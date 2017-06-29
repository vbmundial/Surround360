/**
* Copyright (c) 2016-present, Facebook, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE_render file in the root directory of this subproject. An additional grant
* of patent rights can be found in the PATENTS file in the same directory.
*/

#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <numeric>

#include "Camera.h"
#include "CvUtil.h"
#include "Filter.h"
#include "ImageWarper.h"
#include "MathUtil.h"
#include "MonotonicTable.h"
#include "NovelView.h"
#include "OpticalFlowFactory.h"
#include "RigDescription.h"
#include "SystemUtil.h"
#include "VrCamException.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>

using namespace cv;
using namespace std;
using namespace surround360;
using namespace surround360::calibration;
using namespace surround360::math_util;
using namespace surround360::optical_flow;
using namespace surround360::util;
using namespace surround360::warper;

DEFINE_string(rig_json_file,              "",             "path to json file drescribing camera array");
DEFINE_string(imgs_dir,                   "",             "path to folder of images with names matching cameras in the rig file");
DEFINE_string(frame_number,               "",             "frame number (6-digit zero-padded)");
DEFINE_string(output_data_dir,            "",             "path to write spherical projections for debugging");
DEFINE_string(prev_frame_data_dir,        "NONE",         "path to data for previous frame; used for temporal regularization");
DEFINE_string(output_cubemap_path,        "",             "path to write output oculus 360 cubemap");
DEFINE_string(output_equirect_path,       "",             "path to write output oculus 360 cubemap");
DEFINE_double(interpupilary_dist,         6.4,            "separation of eyes for stereo, spherical_in whatever units the rig json uses.");
DEFINE_int32(side_alpha_feather_size,     101,            "alpha feather for projection of side cameras to spherical coordinates");
DEFINE_int32(std_alpha_feather_size,      31,             "alpha feather for all other purposes. must be odd");
DEFINE_bool(save_debug_images,            false,          "if true, lots of debug images are generated");
DEFINE_double(sharpenning,                0.0f,           "0.0 to 1.0 amount of sharpenning");
DEFINE_bool(enable_top,                   false,          "is there a top camera?");
DEFINE_bool(enable_bottom,                false,          "are there two bottom cameras?");
DEFINE_bool(enable_pole_removal,          false,          "if true, pole removal masks are used; if false, primary bottom camera is used");
DEFINE_string(bottom_pole_masks_dir,      "",             "path to bottom camera pole masks dir");
DEFINE_bool(enable_exposure_comp,         false,          "do exposure compensation?");
DEFINE_int32(exposure_comp_block_size,    0,             "exposure compensation block size for block-gain based correction, 0 for auto");
DEFINE_string(side_flow_alg,              "pixflow_low",  "which optical flow algorithm to use for sides");
DEFINE_string(polar_flow_alg,             "pixflow_low",  "which optical flow algorithm to use for top/bottom warp with sides");
DEFINE_string(poleremoval_flow_alg,       "pixflow_low",  "which optical flow algorithm to use for pole removal with secondary bottom camera");
DEFINE_double(zero_parallax_dist,         10000.0,        "distance where parallax is zero");
DEFINE_int32(eqr_width,                   256,            "width of spherical projection image (0 to 2pi)");
DEFINE_int32(eqr_height,                  128,            "height of spherical projection image (0 to pi)");
DEFINE_int32(final_eqr_width,             3480,           "resize before stacking stereo equirect width");
DEFINE_int32(final_eqr_height,            960,            "resize before stacking stereo equirect height");
DEFINE_int32(cubemap_width,               1536,           "face width of output cubemaps");
DEFINE_int32(cubemap_height,              1536,           "face height of output cubemaps");
DEFINE_string(cubemap_format,             "video",        "either video or photo");

const Camera::Vector3 kGlobalUp = Camera::Vector3::UnitZ();

// measured in radians from forward
float approximateFov(const Camera& camera, const bool vertical) {
  Camera::Vector2 a = camera.principal;
  Camera::Vector2 b = camera.principal;
  if (vertical) {
    a.y() = 0;
    b.y() = camera.resolution.y();
  } else {
    a.x() = 0;
    b.x() = camera.resolution.x();
  }
  return acos(max(
    camera.rig(a).direction().dot(camera.forward()),
    camera.rig(b).direction().dot(camera.forward())));
}

// measured in radians from forward
float approximateFov(const Camera::Rig& rig, const bool vertical) {
  float result = 0;
  for (const auto& camera : rig) {
    result = std::max(result, approximateFov(camera, vertical));
  }
  return result;
}

void projectSideToSpherical(
    Mat& dst,
    const Mat& src,
    const Camera& camera,
    const float leftAngle,
    const float rightAngle,
    const float topAngle,
    const float bottomAngle) {

  // convert, clone or reference, as needed
  Mat tmp = src;
  if (src.channels() == 3) {
    cvtColor(src, tmp, CV_BGR2BGRA);
  } else if (FLAGS_side_alpha_feather_size) {
    tmp = src.clone();
  }
  // feather
  if (FLAGS_side_alpha_feather_size) {
    tmp = featherAlphaChannel(tmp, FLAGS_side_alpha_feather_size);
  }
  // remap
  bicubicRemapToSpherical(
    dst,
    tmp,
    camera,
    leftAngle,
    rightAngle,
    topAngle,
    bottomAngle);
}

// project all of the (side) cameras' images into spherical coordinates
void projectSphericalCamImages(
      const RigDescription& rig,
      const string& imagesDir,
      const string& frameNumber,
      vector<Mat>& projectionImages) {

  VLOG(1) << "Projecting side camera images to spherical coordinates";

  const double startLoadCameraImagesTime = getCurrTimeSec();
  vector<Mat> camImages = rig.loadSideCameraImages(imagesDir, frameNumber);
  const double endLoadCameraImagesTime = getCurrTimeSec();
  VLOG(1) << "Time to load images from file: "
    << endLoadCameraImagesTime - startLoadCameraImagesTime
    << " sec";

  projectionImages.resize(camImages.size());
  vector<std::thread> threads;
  const float hRadians = 2 * approximateFov(rig.rigSideOnly, false);
  const float vRadians = 2 * approximateFov(rig.rigSideOnly, true);
  for (int camIdx = 0; camIdx < camImages.size(); ++camIdx) {
    const Camera& camera = rig.rigSideOnly[camIdx];
    projectionImages[camIdx].create(
      FLAGS_eqr_height * vRadians / M_PI,
      FLAGS_eqr_width * hRadians / (2 * M_PI),
      CV_8UC4);
    // the negative sign here is so the camera array goes clockwise
    float direction = -float(camIdx) / float(camImages.size()) * 2.0f * M_PI;
    threads.emplace_back(
      projectSideToSpherical,
      ref(projectionImages[camIdx]),
      cref(camImages[camIdx]),
      cref(camera),
      direction + hRadians / 2,
      direction - hRadians / 2,
      vRadians / 2,
      -vRadians / 2);
  }
  for (std::thread& t : threads) { t.join(); }

  if (FLAGS_save_debug_images) {
    const string projectionsDir =
      FLAGS_output_data_dir + "/debug/" + FLAGS_frame_number + "/projections/";
    for (int camIdx = 0; camIdx < rig.getSideCameraCount(); ++camIdx) {
      const string cropImageFilename = projectionsDir +
        "/crop_" + rig.getSideCameraId(camIdx) + ".tif";
      imwriteExceptionOnFail(cropImageFilename, projectionImages[camIdx]);
    }
  }
}

// this is where the main work of optical flow for adjacent side cameras is done
void prepareNovelViewGeneratorThread(
    const int overlapImageWidth,
    const int leftIdx, // only used to determine debug image filename
    Mat* imageL,
    Mat* imageR,
    NovelViewGenerator* novelViewGen) {

  Mat overlapImageL = (*imageL)(Rect(
    imageL->cols - overlapImageWidth, 0, overlapImageWidth, imageL->rows));
  Mat overlapImageR = (*imageR)(Rect(0, 0, overlapImageWidth, imageR->rows));

  // save the images that are going into flow. we will need them in the next frame
  if (FLAGS_cubemap_format == "video") {
    const string flowImagesDir =
      FLAGS_output_data_dir + "/debug/" + FLAGS_frame_number + "/flow_images/";
    imwriteExceptionOnFail(
      flowImagesDir + "/overlap_" + std::to_string(leftIdx) + "_L.tif",
      overlapImageL);
    imwriteExceptionOnFail(
      flowImagesDir + "/overlap_" + std::to_string(leftIdx) + "_R.tif",
      overlapImageR);
  }

  // read the previous frame's flow results, if available
  Mat prevFrameFlowLtoR;
  Mat prevFrameFlowRtoL;
  Mat prevOverlapImageL;
  Mat prevOverlapImageR;
  if (FLAGS_prev_frame_data_dir != "NONE") {
    VLOG(1) << "Reading previous frame flow and images from: "
      << FLAGS_prev_frame_data_dir;

    const string flowPrevDir =
      FLAGS_output_data_dir + "/flow/" + FLAGS_prev_frame_data_dir;
    const string flowImagesPrevDir =
      FLAGS_output_data_dir + "/debug/" + FLAGS_prev_frame_data_dir + "/flow_images/";

    prevFrameFlowLtoR = readFlowFromFile(
      flowPrevDir + "/flowLtoR_" + std::to_string(leftIdx) + ".bin");
    prevFrameFlowRtoL = readFlowFromFile(
      flowPrevDir + "/flowRtoL_" + std::to_string(leftIdx) + ".bin");
    prevOverlapImageL = imreadExceptionOnFail(
      flowImagesPrevDir + "/overlap_" + std::to_string(leftIdx) + "_L.tif",
      -1);
    prevOverlapImageR = imreadExceptionOnFail(
      flowImagesPrevDir + "/overlap_" + std::to_string(leftIdx) + "_R.tif",
      -1);
    VLOG(1) << "Loaded previous frame's flow OK";
  }

  // this is the call to actually compute optical flow
  novelViewGen->prepare(
    overlapImageL,
    overlapImageR,
    prevFrameFlowLtoR,
    prevFrameFlowRtoL,
    prevOverlapImageL,
    prevOverlapImageR);

  // get the results of flow and save them. we will need these for temporal regularization
  if (FLAGS_cubemap_format == "video") {
    const Mat flowLtoR = novelViewGen->getFlowLtoR();
    const Mat flowRtoL = novelViewGen->getFlowRtoL();
    const string flowDir = FLAGS_output_data_dir + "/flow/" + FLAGS_frame_number;
    saveFlowToFile(
      flowLtoR,
      flowDir + "/flowLtoR_" + std::to_string(leftIdx) + ".bin");
    saveFlowToFile(
      flowRtoL,
      flowDir + "/flowRtoL_" + std::to_string(leftIdx) + ".bin");
  }
}

// a "chunk" is the portion from a pair of overlapping cameras. returns left/right images
void renderStereoPanoramaChunksThread(
    const int leftIdx, // left camera
    const int numCams,
    const int camImageWidth,
    const int camImageHeight,
    const int numNovelViews,
    const float fovHorizontalRadians,
    const float vergeAtInfinitySlabDisplacement,
    NovelViewGenerator* novelViewGen,
    Mat* chunkL,
    Mat* chunkR) {

  int currChunkX = 0; // current column in chunk to write
  LazyNovelViewBuffer lazyNovelViewBuffer(FLAGS_eqr_width / numCams, camImageHeight);
  for (int nvIdx = 0; nvIdx < numNovelViews; ++nvIdx) {
    const float shift = float(nvIdx) / float(numNovelViews);
    const float slabShift =
      float(camImageWidth) * 0.5f - float(numNovelViews - nvIdx);

    for (int v = 0; v < camImageHeight; ++v) {
      lazyNovelViewBuffer.warpL[currChunkX][v] =
        Point3f(slabShift + vergeAtInfinitySlabDisplacement, v, shift);
      lazyNovelViewBuffer.warpR[currChunkX][v] =
        Point3f(slabShift - vergeAtInfinitySlabDisplacement, v, shift);
    }
    ++currChunkX;
  }

  const int rightIdx = (leftIdx + 1) % numCams;
  pair<Mat, Mat> lazyNovelChunksLR =
    novelViewGen->combineLazyNovelViews(lazyNovelViewBuffer);
  *chunkL = lazyNovelChunksLR.first;
  *chunkR = lazyNovelChunksLR.second;
}

// generates a left/right eye equirect panorama using slices of novel views
void generateRingOfNovelViewsAndRenderStereoSpherical(
    const float cameraRingRadius,
    const float camFovHorizontalDegrees,
    vector<Mat>& projectionImages,
    Mat& panoImageL,
    Mat& panoImageR,
    double& opticalFlowRuntime,
    double& novelViewRuntime) {

  const int numCams = projectionImages.size();

  // this is the amount of horizontal overlap the cameras would have if they
  // were all perfectly aligned (in fact due to misalignment they overlap by a
  // different amount for each pair, but we ignore that to make it simple)
  const float fovHorizontalRadians = toRadians(camFovHorizontalDegrees);
  const float overlapAngleDegrees =
    (camFovHorizontalDegrees * float(numCams) - 360.0) / float(numCams);
  const int camImageWidth = projectionImages[0].cols;
  const int camImageHeight = projectionImages[0].rows;
  const int overlapImageWidth =
    float(camImageWidth) * (overlapAngleDegrees / camFovHorizontalDegrees);
  const int numNovelViews = camImageWidth - overlapImageWidth; // per image pair

  // setup parallel optical flow
  double startOpticalFlowTime = getCurrTimeSec();
  vector<NovelViewGenerator*> novelViewGenerators(projectionImages.size());
  vector<std::thread> threads;
  for (int leftIdx = 0; leftIdx < projectionImages.size(); ++leftIdx) {
    const int rightIdx = (leftIdx + 1) % projectionImages.size();
    novelViewGenerators[leftIdx] =
      new NovelViewGeneratorAsymmetricFlow(FLAGS_side_flow_alg);
    threads.push_back(std::thread(
      prepareNovelViewGeneratorThread,
      overlapImageWidth,
      leftIdx,
      &projectionImages[leftIdx],
      &projectionImages[rightIdx],
      novelViewGenerators[leftIdx]
    ));
  }
  for (std::thread& t : threads) { t.join(); }

  opticalFlowRuntime = getCurrTimeSec() - startOpticalFlowTime;

  // lightfield/parallax formulas
  const float v =
    atanf(FLAGS_zero_parallax_dist / (FLAGS_interpupilary_dist / 2.0f));
  const float psi =
    asinf(sinf(v) * (FLAGS_interpupilary_dist / 2.0f) / cameraRingRadius);
  const float vergeAtInfinitySlabDisplacement =
    psi * (float(camImageWidth) / fovHorizontalRadians);
  const float theta = -M_PI / 2.0f + v + psi;
  const float zeroParallaxNovelViewShiftPixels =
    float(FLAGS_eqr_width) * (theta / (2.0f * M_PI));

  double startNovelViewTime = getCurrTimeSec();
  // a "chunk" will be just the part of the panorama formed from one pair of
  // adjacent cameras. we will stack them horizontally to build the full
  // panorama. we do this so it can be parallelized.
  vector<Mat> panoChunksL(projectionImages.size(), Mat());
  vector<Mat> panoChunksR(projectionImages.size(), Mat());
  vector<std::thread> panoThreads;
  for (int leftIdx = 0; leftIdx < projectionImages.size(); ++leftIdx) {
    panoThreads.push_back(std::thread(
      renderStereoPanoramaChunksThread,
      leftIdx,
      numCams,
      camImageWidth,
      camImageHeight,
      numNovelViews,
      fovHorizontalRadians,
      vergeAtInfinitySlabDisplacement,
      novelViewGenerators[leftIdx],
      &panoChunksL[leftIdx],
      &panoChunksR[leftIdx]
    ));
  }
  for (std::thread& t : panoThreads) { t.join(); }

  novelViewRuntime = getCurrTimeSec() - startNovelViewTime;

  for (int leftIdx = 0; leftIdx < projectionImages.size(); ++leftIdx) {
    delete novelViewGenerators[leftIdx];
  }

  panoImageL = stackHorizontal(panoChunksL);
  if (FLAGS_interpupilary_dist != 0)
    panoImageR = stackHorizontal(panoChunksR);

  panoImageL = offsetHorizontalWrap(panoImageL, zeroParallaxNovelViewShiftPixels);
  if (FLAGS_interpupilary_dist != 0)
    panoImageR = offsetHorizontalWrap(panoImageR, -zeroParallaxNovelViewShiftPixels);
}

// handles flow between the fisheye top or bottom with the left/right eye side panoramas
void poleToSideFlowThread(
    string eyeName,
    const RigDescription& rig,
    Mat* sideSphericalForEye,
    Mat* fisheyeSpherical,
    Mat* warpedSphericalForEye) {

  // crop the side panorama to the height of the pole image
  Mat croppedSideSpherical = (*sideSphericalForEye)(Rect(0, 0, fisheyeSpherical->cols, fisheyeSpherical->rows));
  croppedSideSpherical = featherAlphaChannel(croppedSideSpherical, FLAGS_std_alpha_feather_size);

  // extend the panoramas and wrap horizontally so we can avoid a seam
  const float kExtendFrac = 1.2f;
  const int extendedWidth = float(fisheyeSpherical->cols) * kExtendFrac;
  Mat extendedSideSpherical(Size(extendedWidth, fisheyeSpherical->rows), CV_8UC4);
  Mat extendedFisheyeSpherical(extendedSideSpherical.size(),  CV_8UC4);
  for (int y = 0; y < extendedSideSpherical.rows; ++y) {
    for (int x = 0; x < extendedSideSpherical.cols; ++x) {
      extendedSideSpherical.at<Vec4b>(y, x) =
        croppedSideSpherical.at<Vec4b>(y, x % fisheyeSpherical->cols);
      extendedFisheyeSpherical.at<Vec4b>(y, x) =
        fisheyeSpherical->at<Vec4b>(y, x % fisheyeSpherical->cols);
    }
  }

  if (FLAGS_save_debug_images) {
    const string flowImagesDir =
      FLAGS_output_data_dir + "/debug/" + FLAGS_frame_number + "/flow_images/";
    imwriteExceptionOnFail(flowImagesDir + "/extendedSideSpherical_" + eyeName + ".tif", extendedSideSpherical);
    imwriteExceptionOnFail(flowImagesDir + "/extendedFisheyeSpherical_" + eyeName + ".tif", extendedFisheyeSpherical);
  }

  Mat prevFisheyeFlow;
  Mat prevExtendedSideSpherical;
  Mat prevExtendedFisheyeSpherical;
  if (FLAGS_prev_frame_data_dir != "NONE") {
    VLOG(1) << "Reading previous frame fisheye flow results from: "
      << FLAGS_prev_frame_data_dir;

    const string flowPrevDir =
      FLAGS_output_data_dir + "/flow/" + FLAGS_prev_frame_data_dir;
    const string flowImagesPrevDir =
      FLAGS_output_data_dir + "/debug/" + FLAGS_prev_frame_data_dir + "/flow_images/";

    prevFisheyeFlow = readFlowFromFile(
      flowPrevDir + "/flow_" + eyeName + ".bin");
    prevExtendedSideSpherical = imreadExceptionOnFail(
      flowImagesPrevDir + "/extendedSideSpherical_" + eyeName + ".tif", -1);
    prevExtendedFisheyeSpherical = imreadExceptionOnFail(
      flowImagesPrevDir + "/extendedFisheyeSpherical_" + eyeName + ".tif", -1);
  }

  Mat flow;
  OpticalFlowInterface* flowAlg = makeOpticalFlowByName(FLAGS_polar_flow_alg);
  flowAlg->computeOpticalFlow(
    extendedSideSpherical,
    extendedFisheyeSpherical,
    prevFisheyeFlow,
    prevExtendedSideSpherical,
    prevExtendedFisheyeSpherical,
    flow,
    OpticalFlowInterface::DirectionHint::DOWN);
  delete flowAlg;

  if (FLAGS_cubemap_format == "video") {
    VLOG(1) << "Serializing fisheye flow result";
    const string flowDir = FLAGS_output_data_dir + "/flow/" + FLAGS_frame_number;
    saveFlowToFile(flow, flowDir + "/flow_" + eyeName + ".bin");
  }
  // make a ramp for alpha/flow magnitude
  //const float kRampFrac = 1.0f; // fraction of available overlap used for ramp
  const float kRampFrac = 1.0f; // test: fraction of available overlap used for ramp

  // use fov from bottom camera
  float poleCameraRadius = rig.findCameraByDirection(-kGlobalUp).getFov();

  // use fov from first side camera
  float sideCameraRadius = approximateFov(rig.rigSideOnly, true);

  // crop is average of side and pole cameras
  /*float poleCameraCropRadius = 0.5f * (M_PI / 2 - sideCameraRadius) +
      0.5f * (std::min(float(M_PI / 2), poleCameraRadius))*/;
  // test: weighted average
  float weigthSide = 0.5f;
  float poleCameraCropRadius = weigthSide * (M_PI / 2 - sideCameraRadius) +
    (1.0f - weigthSide) * (std::min(float(M_PI / 2), poleCameraRadius));

  // convert from radians to degrees
  poleCameraCropRadius *= 180 / M_PI;
  poleCameraRadius *= 180 / M_PI;
  sideCameraRadius *= 180 / M_PI;

  const float phiFromPole = poleCameraCropRadius;
  const float phiFromSide = 90.0f - sideCameraRadius;
  const float phiMid = (phiFromPole + phiFromSide) / 2.0f;
  const float phiDiff = fabsf(phiFromPole - phiFromSide);
  const float phiRampStart = phiMid - kRampFrac * phiDiff / 2.0f;
  const float phiRampEnd = phiMid + kRampFrac * phiDiff / 2.0f;

  // ramp for flow magnitude
  //    1               for phi from 0 to phiRampStart
  //    linear drop-off for phi from phiRampStart to phiMid
  //    0               for phi from phiMid to totalRadius
  Mat warp(extendedFisheyeSpherical.size(), CV_32FC2);
  for (int y = 0; y < warp.rows; ++y) {
    const float phi = poleCameraRadius * float(y + 0.5f) / float(warp.rows);
    const float alpha = 1.0f - rampf(phi, phiRampStart, phiMid);
    for (int x = 0; x < warp.cols; ++x) {
      warp.at<Point2f>(y, x) = Point2f(x, y) + (1.0f - alpha) * flow.at<Point2f>(y, x);
    }
  }

  Mat warpedExtendedFisheyeSpherical;
  remap(
    extendedFisheyeSpherical,
    warpedExtendedFisheyeSpherical,
    warp,
    Mat(),
    CV_INTER_CUBIC,
    BORDER_CONSTANT);

  // take the extra strip on the right side and alpha-blend it out on the left side of the result
  *warpedSphericalForEye = warpedExtendedFisheyeSpherical(Rect(0, 0, fisheyeSpherical->cols, fisheyeSpherical->rows));
  int maxBlendX = float(fisheyeSpherical->cols) * (kExtendFrac - 1.0f);
  for (int y = 0; y < warpedSphericalForEye->rows; ++y) {
    for (int x = 0; x < maxBlendX; ++x) {
      const float srcB = warpedSphericalForEye->at<Vec4b>(y, x)[0];
      const float srcG = warpedSphericalForEye->at<Vec4b>(y, x)[1];
      const float srcR = warpedSphericalForEye->at<Vec4b>(y, x)[2];
      const float srcA = warpedSphericalForEye->at<Vec4b>(y, x)[3];
      const float wrapB = warpedExtendedFisheyeSpherical.at<Vec4b>(y, x + fisheyeSpherical->cols)[0];
      const float wrapG = warpedExtendedFisheyeSpherical.at<Vec4b>(y, x + fisheyeSpherical->cols)[1];
      const float wrapR = warpedExtendedFisheyeSpherical.at<Vec4b>(y, x + fisheyeSpherical->cols)[2];
      float alpha = 1.0f - rampf(x, float(maxBlendX) * 0.333f, float(maxBlendX) * 0.667f);
      warpedSphericalForEye->at<Vec4b>(y, x) = Vec4b(
        wrapB * alpha + srcB * (1.0f - alpha),
        wrapG * alpha + srcG * (1.0f - alpha),
        wrapR * alpha + srcR * (1.0f - alpha),
        srcA);
    }
  }

  // make a ramp in the alpha channel for blending with the sides
  //    1               for phi from 0 to phiMid
  //    linear drop-off for phi from phiMid to phiRampEnd
  //    0               for phi from phiRampEnd to totalRadius
  for (int y = 0; y < warp.rows; ++y) {
    const float phi = poleCameraRadius * float(y + 0.5f) / float(warp.rows);
    const float alpha = 1.0f - rampf(phi, phiMid, phiRampEnd);
    for (int x = 0; x < warp.cols; ++x) {
      (*warpedSphericalForEye).at<Vec4b>(y, x)[3] *= alpha;
    }
  }

  copyMakeBorder(
    *warpedSphericalForEye,
    *warpedSphericalForEye,
    0,
    sideSphericalForEye->rows - warpedSphericalForEye->rows,
    0,
    0,
    BORDER_CONSTANT,
    Scalar(0,0,0,0));

  if (FLAGS_save_debug_images) {
    const string debugDir =
      FLAGS_output_data_dir + "/debug/" + FLAGS_frame_number;
    imwriteExceptionOnFail(
      debugDir + "/croppedSideSpherical_" + eyeName + ".tif",
      croppedSideSpherical);
    imwriteExceptionOnFail(
      debugDir + "/warpedSpherical_" + eyeName + ".tif",
      *warpedSphericalForEye);
    imwriteExceptionOnFail(
      debugDir + "/extendedSideSpherical_" + eyeName + ".tif",
      extendedSideSpherical);
  }
}

// does pole removal from the two bottom cameras, and projects the result to equirect
void prepareBottomImagesThread(
    const RigDescription& rig,
    vector<Mat>& bottomSphericals) {

  bottomSphericals.resize(rig.rigBottomOnly.size());
  for (int i = 0; i < rig.rigBottomOnly.size(); ++i) {
    Mat& bottomSpherical = bottomSphericals[i];

    const string cameraDir = FLAGS_imgs_dir + "/" + rig.rigBottomOnly[i].id;
    const string bottomImagePath = cameraDir + "/" + FLAGS_frame_number + ".tif";
    Mat bottomImage = imreadExceptionOnFail(bottomImagePath, CV_LOAD_IMAGE_UNCHANGED);
    if (FLAGS_enable_pole_removal) {
      if (bottomImage.channels() != 4) {
        cvtColor(bottomImage, bottomImage, CV_BGR2BGRA);
        const string poleMaskPath = FLAGS_bottom_pole_masks_dir + "/" 
                                  + rig.rigBottomOnly[i].id + ".tif";
        Mat bottomRedMask = imreadExceptionOnFail(poleMaskPath, CV_LOAD_IMAGE_COLOR);
        cutRedMaskOutOfAlphaChannel(bottomImage, bottomRedMask);
      }
    }

    const Camera& camera = rig.rigBottomOnly[i];
    bottomSpherical.create(
      FLAGS_eqr_height * camera.getFov() / M_PI,
      FLAGS_eqr_width,
      CV_8UC4);
    bicubicRemapToSpherical(
      bottomSpherical,
      bottomImage,
      camera,
      0,
      2.0f * M_PI,
      -(M_PI / 2.0f),
      -(M_PI / 2.0f - camera.getFov()));

    // Remove upper half of projected image (it is flipped, so lower half)
    Size size = bottomSpherical.size();
    bottomSpherical(Rect(0, size.height / 2, size.width, size.height / 2)) = Vec4b(0,0,0,0);

    // the alpha channel in bottomSpherical is the result of pole removal/flow. this can in
    // some cases cause an alpha-channel discontinuity at the boundary of the image, which
    // will have an effect on flow between bottom and sides. to mitigate that, we do another
    // pass of feathering on bottomSpherical before converting to polar coordinates.
    bottomSpherical = featherAlphaChannel(bottomSpherical, FLAGS_std_alpha_feather_size);

    if (FLAGS_save_debug_images) {
      const string debugDir =
        FLAGS_output_data_dir + "/debug/" + FLAGS_frame_number;
      imwriteExceptionOnFail(debugDir + "/" + camera.id + "_bottomSpherical.tif", bottomSpherical);
    }
  }
}

// similar to prepareBottomImagesThread but there is no pole removal
void prepareTopImagesThread(const RigDescription& rig, vector<Mat>& topSphericals) {

  topSphericals.resize(rig.rigTopOnly.size());
  for (int i = 0; i < rig.rigTopOnly.size(); ++i) {
    Mat& topSpherical = topSphericals[i];
    const string cameraDir = FLAGS_imgs_dir + "/" + rig.rigTopOnly[i].id;
    const string topImageFilename = FLAGS_frame_number + ".tif";
    const string topImagePath = cameraDir + "/" + topImageFilename;
    Mat topImage = imreadExceptionOnFail(topImagePath, CV_LOAD_IMAGE_UNCHANGED);
    const Camera& camera = rig.rigTopOnly[i];
    topSpherical.create(
      FLAGS_eqr_height * camera.getFov() / M_PI,
      FLAGS_eqr_width,
      CV_8UC4);
    bicubicRemapToSpherical(
      topSpherical,
      topImage,
      camera,
      2.0f * M_PI,
      0,
      M_PI / 2.0f,
      M_PI / 2.0f - camera.getFov());

    // alpha feather the top spherical image for flow purposes
    if (topSpherical.type() != CV_8UC4) {
      cvtColor(topSpherical, topSpherical, CV_BGR2BGRA);
    }

    // Remove lower half of projected image
    Size size = topSpherical.size();
    topSpherical(Rect(0, size.height / 2, size.width, size.height / 2)) = Vec4b(0, 0, 0, 0);

    topSpherical = featherAlphaChannel(topSpherical, FLAGS_std_alpha_feather_size);

    if (FLAGS_save_debug_images) {
      const string debugDir =
        FLAGS_output_data_dir + "/debug/" + FLAGS_frame_number;
      imwriteExceptionOnFail(debugDir + "/" + camera.id + "_topSpherical.tif", topSpherical);
    }
  }
}

// sharpen the left or right eye panorama using a periodic boundary
void sharpenThread(Mat* sphericalImage) {
  const WrapBoundary<float> wrapB;
  const ReflectBoundary<float> reflectB;
  Mat lowPassSphericalImage(sphericalImage->rows, sphericalImage->cols, CV_8UC3);
  iirLowPass<WrapBoundary<float>, ReflectBoundary<float>, Vec3b>(
    *sphericalImage, 0.25f, lowPassSphericalImage, wrapB, reflectB);
  sharpenWithIirLowPass<Vec3b>(
    *sphericalImage, lowPassSphericalImage, 1.0f + FLAGS_sharpenning);
}

// If the un-padded height is odd and targetHeight is even, we can't do equal
// padding to get the final image to be targetHeight. the formulas below give
// equal padding if possible, or equal +/-1 if not.
void padToheight(Mat& unpaddedImage, const int targetHeight) {
  const int paddingAbove = (targetHeight - unpaddedImage.rows) / 2;
  const int paddingBelow = targetHeight - unpaddedImage.rows - paddingAbove;
  copyMakeBorder(
    unpaddedImage,
    unpaddedImage,
    paddingAbove,
    paddingBelow,
    0,
    0,
    BORDER_CONSTANT,
    Scalar(0.0, 0.0, 0.0));
}

void splitAlphaMask(Mat& image, Mat& mask)
{
  CV_Assert(image.type() == CV_8UC4);
  vector<Mat> channels(4);
  split(image, channels);
  mask = channels.back();

  channels.pop_back();
  merge(channels, image);
}

void mergeAlphaMask(Mat& image, const Mat& mask)
{
  CV_Assert(image.type() == CV_8UC3 && mask.type() == CV_8UC1);
  // add alpha channel
  vector<Mat> channels(3);
  split(image, channels);
  channels.push_back(mask);
  merge(channels, image);
}

// run the whole stereo panorama rendering pipeline
void renderStereoPanorama() {
  requireArg(FLAGS_rig_json_file, "rig_json_file");
  requireArg(FLAGS_imgs_dir, "imgs_dir");
  requireArg(FLAGS_frame_number, "frame_number");
  requireArg(FLAGS_output_data_dir, "output_data_dir");
  requireArg(FLAGS_output_equirect_path, "output_equirect_path");

  const double startTime = getCurrTimeSec();

  const string debugDir =
    FLAGS_output_data_dir + "/debug/" + FLAGS_frame_number;

  RigDescription rig(FLAGS_rig_json_file);
  if (FLAGS_eqr_width % rig.getSideCameraCount() != 0) {
    VLOG(1) << "Number of side cameras:" << rig.getSideCameraCount();
    VLOG(1) << "Suggested widths:";
    for (int i = FLAGS_eqr_width * 0.9; i < FLAGS_eqr_width * 1.1; ++i) {
      if (i % rig.getSideCameraCount() == 0) {
        VLOG(1) << i;
      }
    }
    throw VrCamException("eqr_width must be evenly divisible by the number of cameras");
  }

  // prepare the bottom camera(s) by doing pole removal and projections in a thread.
  // will join that thread as late as possible.
  vector<Mat> bottomSphericals;
  thread prepareBottomThread;
  if (FLAGS_enable_bottom) {
    VLOG(1) << "Bottom cameras enabled. Preparing bottom projections in a thread";
    prepareBottomThread = std::thread(
      prepareBottomImagesThread, std::cref(rig), std::ref(bottomSphericals));
  }

  // top cameras are handled similar to bottom cameras- do anything we can in a thread
  // that is joined as late as possible.
  vector<Mat> topSphericals;
  std::thread prepareTopThread;
  if (FLAGS_enable_top) {
    prepareTopThread = std::thread(
      prepareTopImagesThread, cref(rig), std::ref(topSphericals));
  }

  // projection to spherical coordinates
  vector<Mat> projectionImages;

  if (FLAGS_save_debug_images) {
    const string projectionsDir =
      FLAGS_output_data_dir + "/debug/" + FLAGS_frame_number + "/projections/";
    system(string("rm -f " + projectionsDir + "/*").c_str());
  }

  const double startProjectSphericalTime = getCurrTimeSec();
  LOG(INFO) << "Projecting camera images to spherical";
  projectSphericalCamImages(rig, FLAGS_imgs_dir, FLAGS_frame_number, projectionImages);
  const double endProjectSphericalTime = getCurrTimeSec();

  // Do exposure compensation
  if (FLAGS_enable_exposure_comp) {
    if (FLAGS_enable_top) {
      prepareTopThread.join(); // this is the latest we can wait
    }
    if (FLAGS_enable_bottom) {
      prepareBottomThread.join(); // this is the latest we can wait
    }

    vector<Mat*> images;
    vector<Point> corners;
    int stripWidth = FLAGS_eqr_width / projectionImages.size();
    int sideImagesYcorner = FLAGS_eqr_height / 2 - projectionImages[0].size().height / 2;
    for (int i = 0; i < projectionImages.size(); ++i) {
      images.push_back(&projectionImages[i]);
      corners.push_back({ i * stripWidth, sideImagesYcorner });
    }
    for (int i = 0; i < topSphericals.size(); ++i) {
      images.push_back(&topSphericals[i]);
      corners.push_back({ 0, 0 });
    }
    for (int i = 0; i < bottomSphericals.size(); ++i) {
      flip(bottomSphericals[i], bottomSphericals[i], -1);
      images.push_back(&bottomSphericals[i]);
      corners.push_back({ 0, 0 });
    }
    int imageCount = images.size();

    vector<Mat> masks;
    vector<pair<UMat, uchar>> maskUMatPairs;
    vector<UMat> imageUMats;
    for (Mat * image : images) {
      Mat mask;
      splitAlphaMask(*image, mask);
      masks.push_back(move(mask));
      maskUMatPairs.push_back({ masks.back().getUMat(ACCESS_READ), 255 });
      imageUMats.push_back(image->getUMat(ACCESS_RW));
    }

    // Compensate for 360 overlap
    imageUMats.reserve(3 * imageCount);
    maskUMatPairs.reserve(3 * imageCount);
    corners.reserve(3 * imageCount);
    for (int i = 0; i < 2; ++i) {
      copy_n(imageUMats.begin(), imageCount, back_inserter(imageUMats));
      copy_n(maskUMatPairs.begin(), imageCount, back_inserter(maskUMatPairs));
      copy_n(corners.begin(), imageCount, back_inserter(corners));  
    }
    
    for (int i = 0; i < imageCount; ++i) {
      corners[imageCount + i].x += FLAGS_eqr_width;
      corners[2 * imageCount + i].x -= FLAGS_eqr_width;
    }

    int blockSize = FLAGS_exposure_comp_block_size;
    if (FLAGS_exposure_comp_block_size <= 0) {
      blockSize = FLAGS_eqr_width / 16;
    }
    BlocksGainCompensator compensator(blockSize, blockSize);
    compensator.feed(corners, imageUMats, maskUMatPairs);

    for (int i = 0; i < imageCount; ++i) {
      compensator.apply(i, corners[i], imageUMats[i], maskUMatPairs[i].first);
      mergeAlphaMask(*images[i], masks[i]);
    }

    for (int i = 0; i < bottomSphericals.size(); ++i) {
      flip(bottomSphericals[i], bottomSphericals[i], -1); // flip back
    }
  }

  // generate novel views and stereo spherical panoramas
  double opticalFlowRuntime, novelViewRuntime;
  Mat sphericalImageL, sphericalImageR;
  LOG(INFO) << "Rendering stereo panorama";
  const double fovHorizontal =
    2 * approximateFov(rig.rigSideOnly, false) * (180 / M_PI);
  generateRingOfNovelViewsAndRenderStereoSpherical(
    rig.getRingRadius(),
    fovHorizontal,
    projectionImages,
    sphericalImageL,
    sphericalImageR,
    opticalFlowRuntime,
    novelViewRuntime);

  if (FLAGS_save_debug_images) {
    VLOG(1) << "Offset-warping images for debugging";
    Mat wrapSphericalImageL, wrapSphericalImageR;
    wrapSphericalImageL = offsetHorizontalWrap(sphericalImageL, sphericalImageL.cols/3);
    if (FLAGS_interpupilary_dist != 0)
      wrapSphericalImageR = offsetHorizontalWrap(sphericalImageR, sphericalImageR.cols/3);
    imwriteExceptionOnFail(debugDir + "/sphericalImgL.tif", sphericalImageL);
    if (FLAGS_interpupilary_dist != 0)
      imwriteExceptionOnFail(debugDir + "/sphericalImgR.tif", sphericalImageR);
    imwriteExceptionOnFail(debugDir + "/sphericalImg_offsetwrapL.tif", wrapSphericalImageL);
    if (FLAGS_interpupilary_dist != 0)
      imwriteExceptionOnFail(debugDir + "/sphericalImg_offsetwrapR.tif", wrapSphericalImageR);
  }

  // so far we only operated on the strip that contains the full vertical FOV of
  // the side cameras. before merging those results with top/bottom cameras,
  // we will pad the side images out to be a full 180 degree vertical equirect.
  padToheight(sphericalImageL, FLAGS_eqr_height);
  if (FLAGS_interpupilary_dist != 0)
    padToheight(sphericalImageR, FLAGS_eqr_height);

  const double topBottomToSideStartTime = getCurrTimeSec();
  
  // if we have a top camera, do optical flow with its image and the side camera
  // composite the results
  if (FLAGS_enable_top) {
    if (!FLAGS_enable_exposure_comp) {
      prepareTopThread.join(); // this is the latest we can wait
    }
    for (int i = 0; i < rig.rigTopOnly.size(); ++i) {
      Mat topSphericalWarpedL, topSphericalWarpedR;
      std::thread topFlowThreadL(
        poleToSideFlowThread,
        "top_left" + rig.rigTopOnly[i].id,
        cref(rig),
        &sphericalImageL,
        &topSphericals[i],
        &topSphericalWarpedL);

      std::thread topFlowThreadR;
      if (FLAGS_interpupilary_dist != 0) {
        topFlowThreadR = std::thread(
          poleToSideFlowThread,
          "top_right" + rig.rigTopOnly[i].id,
          cref(rig),
          &sphericalImageR,
          &topSphericals[i],
          &topSphericalWarpedR);
      }

      topFlowThreadL.join();
      sphericalImageL =
        flattenLayersDeghostPreferBase(sphericalImageL, topSphericalWarpedL);
      if (FLAGS_interpupilary_dist != 0) {
        topFlowThreadR.join();
        sphericalImageR =
          flattenLayersDeghostPreferBase(sphericalImageR, topSphericalWarpedR);
      }
    }
  }

  if (FLAGS_enable_bottom) {
    if (!FLAGS_enable_exposure_comp) {
      prepareBottomThread.join(); // this is the latest we can wait
    }

    for (int i = 0; i < rig.rigBottomOnly.size(); ++i) {
      Mat bottomSphericalWarpedL, bottomSphericalWarpedR;
      // flip the side images upside down for bottom flow
      Mat flipSphericalImageL, flipSphericalImageR;

      flip(sphericalImageL, flipSphericalImageL, -1);
      thread bottomFlowThreadL(
        poleToSideFlowThread, 
        "bottom_left" + rig.rigBottomOnly[i].id,
        cref(rig),
        &flipSphericalImageL,
        &bottomSphericals[i],
        &bottomSphericalWarpedL);

      thread bottomFlowThreadR;
      if (FLAGS_interpupilary_dist != 0) {
        flip(sphericalImageR, flipSphericalImageR, -1);
        bottomFlowThreadR = std::thread(
          poleToSideFlowThread,
          "bottom_right" + rig.rigBottomOnly[i].id,
          cref(rig),
          &flipSphericalImageR,
          &bottomSphericals[i],
          &bottomSphericalWarpedR);
      }

      bottomFlowThreadL.join();
      flipSphericalImageL =
        flattenLayersDeghostPreferBase(flipSphericalImageL, bottomSphericalWarpedL);
      
      if (FLAGS_interpupilary_dist != 0) {
        bottomFlowThreadR.join();
        flipSphericalImageR =
          flattenLayersDeghostPreferBase(flipSphericalImageR, bottomSphericalWarpedR);
        flip(flipSphericalImageR, sphericalImageR, -1);
      }
      flip(flipSphericalImageL, sphericalImageL, -1);
    } 
  }
  const double topBottomToSideEndTime = getCurrTimeSec();

  // depending on how things are handled, we might still have an alpha channel.
  // if so, flatten the image to 3 channel
  if (sphericalImageL.type() != CV_8UC3) {
    VLOG(1) << "Flattening from 4 channels to 3 channels";
    cvtColor(sphericalImageL, sphericalImageL, CV_BGRA2BGR);
    if (FLAGS_interpupilary_dist != 0)
      cvtColor(sphericalImageR, sphericalImageR, CV_BGRA2BGR);
  }

  if (FLAGS_save_debug_images) {
    imwriteExceptionOnFail(debugDir + "/eqr_sideL.tif", sphericalImageL);
    if (FLAGS_interpupilary_dist != 0)
      imwriteExceptionOnFail(debugDir + "/eqr_sideR.tif", sphericalImageR);
  }

  const double startSharpenTime = getCurrTimeSec();
  if (FLAGS_sharpenning > 0.0f) {
    VLOG(1) << "Sharpening";
    std::thread sharpenThreadL(sharpenThread, &sphericalImageL);
    std::thread sharpenThreadR;
    if (FLAGS_interpupilary_dist != 0)
      sharpenThreadR = std::thread(sharpenThread, &sphericalImageR);
    sharpenThreadL.join();
    if (FLAGS_interpupilary_dist != 0)
      sharpenThreadR.join();
    if (FLAGS_save_debug_images) {
      imwriteExceptionOnFail(debugDir + "/_eqr_sideL_sharpened.tif", sphericalImageL);
      if (FLAGS_interpupilary_dist != 0)
        imwriteExceptionOnFail(debugDir + "/_eqr_sideR_sharpened.tif", sphericalImageR);
    }
  }
  const double endSharpenTime = getCurrTimeSec();

  // project the horizontal panoramas to cubemaps and composite the top
  const double startCubemapTime = getCurrTimeSec();
  if (FLAGS_cubemap_width > 0 && FLAGS_cubemap_height > 0
      && !FLAGS_output_cubemap_path.empty()) {
    LOG(INFO) << "Generating stereo cubemap";
    Mat cubemapImageL = stackOutputCubemapFaces(
        FLAGS_cubemap_format,
        convertSphericalToCubemapBicubicRemap(
          sphericalImageL,
          M_PI,
          FLAGS_cubemap_width,
          FLAGS_cubemap_height));
    if (FLAGS_interpupilary_dist != 0) {
      Mat cubemapImageR = stackOutputCubemapFaces(
        FLAGS_cubemap_format, convertSphericalToCubemapBicubicRemap(
          sphericalImageR,
          M_PI,
          FLAGS_cubemap_width,
          FLAGS_cubemap_height));
      Mat stereoCubemap = stackVertical(vector<Mat>({ cubemapImageL, cubemapImageR }));
      imwriteExceptionOnFail(FLAGS_output_cubemap_path, stereoCubemap);
    } else {
      imwriteExceptionOnFail(FLAGS_output_cubemap_path, cubemapImageL);
    }
  }
  const double endCubemapTime = getCurrTimeSec();

  if (FLAGS_final_eqr_width != 0 &&
      FLAGS_final_eqr_height != 0 &&
      FLAGS_final_eqr_width != FLAGS_eqr_width &&
      FLAGS_final_eqr_height != FLAGS_eqr_height / 2) {
    VLOG(1) << "Resizing before final equirect stack (for proper video size)";
    resize(
      sphericalImageL,
      sphericalImageL,
      Size(FLAGS_final_eqr_width, FLAGS_final_eqr_height / 2),
      0,
      0,
      INTER_CUBIC);
    if (FLAGS_interpupilary_dist != 0) {
      resize(
        sphericalImageR,
        sphericalImageR,
        Size(FLAGS_final_eqr_width, FLAGS_final_eqr_height / 2),
        0,
        0,
        INTER_CUBIC);
    }
  }

  LOG(INFO) << "Creating stereo equirectangular image";

   if (FLAGS_interpupilary_dist != 0) {
    Mat stereoEquirect = stackVertical(vector<Mat>({ sphericalImageL, sphericalImageR }));
    imwriteExceptionOnFail(FLAGS_output_equirect_path, stereoEquirect);
   } else {
     imwriteExceptionOnFail(FLAGS_output_equirect_path, sphericalImageL);
   }

  const double endTime = getCurrTimeSec();
  VLOG(1) << "--- Runtime breakdown (sec) ---";
  VLOG(1) << "Total:\t\t\t" << endTime - startTime;
  VLOG(1) << "Spherical projection:\t" << endProjectSphericalTime - startProjectSphericalTime;
  VLOG(1) << "Side optical flow:\t\t" << opticalFlowRuntime;
  VLOG(1) << "Novel view panorama:\t" << novelViewRuntime;
  VLOG(1) << "Flow top+bottom with sides:\t" << topBottomToSideEndTime - topBottomToSideStartTime;
  VLOG(1) << "Sharpen:\t\t" << endSharpenTime - startSharpenTime;
  VLOG(1) << "Equirect -> Cubemap:\t" << endCubemapTime - startCubemapTime;
}

int main(int argc, char** argv) {
  initSurround360(argc, argv);
  renderStereoPanorama();
  return EXIT_SUCCESS;
}

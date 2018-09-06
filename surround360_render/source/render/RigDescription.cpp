/**
* Copyright (c) 2016-present, Facebook, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE_render file in the root directory of this subproject. An additional grant
* of patent rights can be found in the PATENTS file in the same directory.
*/

#include "RigDescription.h"
#include <deque>

namespace surround360 {

using namespace cv;
using namespace std;
using namespace surround360::util;

RigDescription::RigDescription(const string& filename) {
  rig = Camera::loadRig(filename);
  for (const Camera& camera : rig) {
    if (camera.group.find("side") != string::npos) {
      rigSideOnly.emplace_back(camera);
    } else if (camera.group.find("top") != string::npos) {
	  rigTopOnly.emplace_back(camera);
	  } else if (camera.group.find("bottom") != string::npos) {
	    rigBottomOnly.emplace_back(camera);
	  }
  }

  // validation
  CHECK_NE(getSideCameraCount(), 0);
}

const Camera& RigDescription::findCameraByDirection(
    const Camera::Vector3& direction,
    const Camera::Real distCamAxisToRigCenterMax) const {
  const Camera* best = nullptr;
  for (const Camera& camera : rig) {
    if (best == nullptr ||
        best->forward().dot(direction) < camera.forward().dot(direction)) {
      if (distCamAxisToRigCenter(camera) <= distCamAxisToRigCenterMax) {
        best = &camera;
      }
    }
  }
  return *CHECK_NOTNULL(best);
}

// find the camera with the largest distance from camera axis to rig center
const Camera& RigDescription::findLargestDistCamAxisToRigCenter() const {
  const Camera* best = &rig.back();
  for (const Camera& camera : rig) {
    if (distCamAxisToRigCenter(camera) > distCamAxisToRigCenter(*best)) {
      best = &camera;
    }
  }
  return *best;
}

string RigDescription::getTopCameraId() const {
  return findCameraByDirection(Camera::Vector3::UnitZ()).id;
}

string RigDescription::getBottomCameraId() const {
  return findCameraByDirection(-Camera::Vector3::UnitZ()).id;
}

string RigDescription::getBottomCamera2Id() const {
  return findLargestDistCamAxisToRigCenter().id;
}

int RigDescription::getSideCameraCount() const {
  return rigSideOnly.size();
}

string RigDescription::getSideCameraId(const int idx) const {
  return rigSideOnly[idx].id;
}

float RigDescription::getRingRadius() const {
	float sqnormsum = 0;
	for (auto& sidecam : rigSideOnly)
		sqnormsum += sidecam.position.squaredNorm();
	return sqrt(sqnormsum / rigSideOnly.size());
}

vector<Mat> RigDescription::loadSideCameraImages(
    const string& imageDir,
    const string& frameNumber, 
    int threadLimit) const {
    VLOG(1) << "loadSideCameraImages spawning threads";
  deque<std::thread> threads;
  vector<Mat> images(getSideCameraCount());
  for (int i = 0; i < getSideCameraCount(); ++i) {
    const string camDir = imageDir + "/" + getSideCameraId(i);
    string extension = getImageFileExtension(camDir);
    const string filename = frameNumber + "." + extension;
    const string imagePath = camDir + "/" + filename;
    VLOG(1) << "imagePath = " << imagePath;
    if (threads.size() >= threadLimit) {
      threads.front().join();
      threads.pop_front();
    }
    threads.emplace_back(
      imreadInStdThread,
      imagePath,
      CV_LOAD_IMAGE_UNCHANGED,
      &(images[i]));
  }
  for (auto& thread : threads) {
    thread.join();
  }

  return images;
}

Mat RigDescription::loadSideCameraImage(
    int cameraIdx,
    const string& imageDir,
    const string& frameNumber) const {
    VLOG(1) << "loadSideCameraImage";

    const string camDir = imageDir + "/" + getSideCameraId(cameraIdx);
    string extension = getImageFileExtension(camDir);
    const string filename = frameNumber + "." + extension;
    const string imagePath = camDir + "/" + filename;
    VLOG(1) << "imagePath = " << imagePath;
    return imread(imagePath, CV_LOAD_IMAGE_UNCHANGED);
}

} // namespace surround360

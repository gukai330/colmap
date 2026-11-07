// Build and run (no CMake target; TESTS_ENABLED is off in the recon360 wheel build):
//   $CXX -std=c++17 -O1 -I src -I <ceres/include> -I <eigen3> recon360/fisheye_jacobian_check.cc \
//       src/colmap/sensor/models.cc -lglog -lgflags -o /tmp/fjc && /tmp/fjc
// Expected: every model PASS at rel 1e-9 and "JACOBIAN TEST: PASS".
// Verifies the recon360 fisheye ImgFromCamWithJac against autodiff of the
// patched templated ImgFromCam, for every perspective fisheye model, theta in
// [0, 170] deg, and CamRayFromImgWithJac past 90 deg against finite differences.
#include "colmap/sensor/models.h"

#include <ceres/jet.h>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace colmap;

template <typename Model>
std::vector<double> Params() {
  std::vector<double> p(Model::num_params, 0.0);
  for (size_t i : Model::focal_length_idxs) p[i] = 520.0;
  p[Model::principal_point_idxs[0]] = 957.5;
  p[Model::principal_point_idxs[1]] = 961.1;
  for (size_t k = 0; k < Model::num_extra_params; ++k)
    p[Model::extra_params_idxs[k]] = ((k % 2) ? -1.0 : 1.0) * 0.01 / (1.0 + k);
  return p;
}

template <typename Model>
int Check(const char* name) {
  constexpr int NP = Model::num_params;
  using J = ceres::Jet<double, 3 + NP>;
  const std::vector<double> p = Params<Model>();
  int fails = 0, n = 0;
  double max_val = 0, max_juvw = 0, max_jp = 0;
  for (double th = 0.0; th <= 170.0; th += 2.5) {
    for (double ph : {0.3, 1.7, -2.4}) {
      for (double scale : {0.5, 3.0, 40.0}) {
        const double t = th * M_PI / 180.0;
        const double u = scale * std::sin(t) * std::cos(ph), v = scale * std::sin(t) * std::sin(ph),
                     w = scale * std::cos(t);
        double x, y, Juvw[6], Jp[2 * NP];
        const bool ok = Model::ImgFromCamWithJac(p.data(), u, v, w, &x, &y, Jp, Juvw);
        J pj[NP], uj(u, 0), vj(v, 1), wj(w, 2), xj, yj;
        for (int i = 0; i < NP; ++i) pj[i] = J(p[i], 3 + i);
        const bool ok_ad = Model::ImgFromCam(pj, uj, vj, wj, &xj, &yj);
        ++n;
        if (ok != ok_ad || !ok) { ++fails; std::printf("  %s th=%.1f ok=%d ad=%d\n", name, th, ok, ok_ad); continue; }
        const double sc = 1.0 / scale;
        const double mag = std::max(1.0, std::abs(xj.a) + std::abs(yj.a));
        max_val = std::max({max_val, std::abs(x - xj.a) / mag, std::abs(y - yj.a) / mag});
        double e_here = 0, v_here = std::max(std::abs(x - xj.a), std::abs(y - yj.a)) / std::max(1.0, std::abs(xj.a) + std::abs(yj.a));
        for (int c = 0; c < 3; ++c) {
          const double ref0 = std::max(1.0, std::abs(xj.v[c]) + std::abs(yj.v[c]));
          e_here = std::max({e_here, std::abs(Juvw[c] - xj.v[c]) / ref0, std::abs(Juvw[3 + c] - yj.v[c]) / ref0});
        }

        for (int c = 0; c < 3; ++c) {
          const double ref = std::max(1.0, std::abs(xj.v[c]) + std::abs(yj.v[c]));
          max_juvw = std::max({max_juvw, std::abs(Juvw[c] - xj.v[c]) / ref, std::abs(Juvw[3 + c] - yj.v[c]) / ref});
        }
        for (int c = 0; c < NP; ++c) {
          const double ref = std::max(1.0, std::abs(xj.v[3 + c]) + std::abs(yj.v[3 + c]));
          max_jp = std::max({max_jp, std::abs(Jp[c] - xj.v[3 + c]) / ref, std::abs(Jp[NP + c] - yj.v[3 + c]) / ref});
        }
        (void)sc;
      }
    }
  }
  const bool pass = fails == 0 && max_val < 1e-9 && max_juvw < 1e-9 && max_jp < 1e-9;
  std::printf("%-28s %s  n=%d fail=%d  rel|dxy|=%.1e  rel|dJ_uvw|=%.1e  rel|dJ_params|=%.1e\n",
              name, pass ? "PASS" : "FAIL", n, fails, max_val, max_juvw, max_jp);
  return pass ? 0 : 1;
}

// CamRayFromImgWithJac past 90 deg: has a value, rz < 0, J_ray matches FD.
int CheckRay() {
  using M = OpenCVFisheyeCameraModel;
  const std::vector<double> p = {520.44, 520.44, 957.54, 961.06, 0.047585, 0.015722, -0.011407, 0.001013};
  int bad = 0;
  for (double th : {45.0, 89.0, 91.0, 95.0, 100.0}) {
    const double t = th * M_PI / 180.0;
    double x, y;
    M::ImgFromCam(p.data(), std::sin(t) * std::cos(0.7), std::sin(t) * std::sin(0.7), std::cos(t), &x, &y);
    const auto ray = CameraModelCamRayFromImg(M::model_id, p, Eigen::Vector2d(x, y));
    Eigen::Matrix2x3d Juvw;
    const auto img = CameraModelImgFromCamWithJac(M::model_id, p, *ray, &Juvw);
    const auto Jray = img ? CamRayFromImgJacobian(*ray, Juvw) : std::nullopt;
    double err = -1;
    if (Jray) {
      const double h = 1e-4;
      Eigen::Matrix3x2d fd;
      for (int c = 0; c < 2; ++c) {
        Eigen::Vector2d d = Eigen::Vector2d::Zero(); d[c] = h;
        fd.col(c) = (*CameraModelCamRayFromImg(M::model_id, p, Eigen::Vector2d(x, y) + d) -
                     *CameraModelCamRayFromImg(M::model_id, p, Eigen::Vector2d(x, y) - d)) / (2 * h);
      }
      err = (fd - *Jray).norm() / fd.norm();
    }
    const bool ok = Jray && err >= 0 && err < 1e-5 && (th < 90 || (*ray)[2] < 0);
    bad += !ok;
    std::printf("CamRayFromImgWithJac th=%5.1f  ray.z=%+.4f  rel|J-FD|=%.1e  %s\n", th,
                ray ? (*ray)[2] : 0.0, err, ok ? "PASS" : "FAIL");
  }
  return bad;
}

int main() {
  int f = 0;
  f += Check<SimpleRadialFisheyeCameraModel>("SIMPLE_RADIAL_FISHEYE");
  f += Check<RadialFisheyeCameraModel>("RADIAL_FISHEYE");
  f += Check<OpenCVFisheyeCameraModel>("OPENCV_FISHEYE");
  f += Check<ThinPrismFisheyeCameraModel>("THIN_PRISM_FISHEYE");
  f += Check<RadTanThinPrismFisheyeModel>("RAD_TAN_THIN_PRISM_FISHEYE");
  f += Check<SimpleFisheyeCameraModel>("SIMPLE_FISHEYE");
  f += Check<FisheyeCameraModel>("FISHEYE");
  f += CheckRay();
  std::printf(f ? "JACOBIAN TEST: FAIL (%d)\n" : "JACOBIAN TEST: PASS\n", f);
  return f ? 1 : 0;
}

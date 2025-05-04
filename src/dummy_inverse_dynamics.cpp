/**
 * *********************************************************
 *
 * @file: dummy_inverse_dynamics.cpp
 * @brief: 加载urdf文件，通过转动惯量等计算关节力矩，并计算末端执行器的正运动学，测试demo，
 * @author: xiaohai（tetrabot）
 * @date: 2025-4-28
 * @version: 1.0
 *
 * ********************************************************
 */

#include <iostream>
#include <Eigen/Geometry> // 添加Eigen几何模块用于变换矩阵

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/kinematics.hpp" // 添加运动学算法头文件
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
// 模型文件所在路径
#ifndef PINOCCHIO_MODEL_DIR
  #define PINOCCHIO_MODEL_DIR "../dummy_description/urdf/"
#endif

int main(int argc, char ** argv)
{
  using namespace pinocchio;

  // URDF文件
  const std::string urdf_filename = (argc<=1) ? PINOCCHIO_MODEL_DIR + std::string("dummy_gripper.urdf") : argv[1];

  // 加载URDF文件
  Model model;
  pinocchio::urdf::buildModel(urdf_filename, model);
  std::cout << "model name: " << model.name << std::endl;
  const int JOINT_ID = 6;

  // 创建数据存储变量，存储算法运行相关的数据
  Data data(model);

  // 给定关节角度、角速度及角加速度
  Eigen::VectorXd q = (Eigen::VectorXd(8) << 0, 0.5, 0, 0, 0, 0, 0, 0).finished();
  Eigen::VectorXd v = Eigen::VectorXd::Zero(model.nv);
  Eigen::VectorXd a = Eigen::VectorXd::Zero(model.nv);

  // 计算关节力矩
  Eigen::VectorXd tau = pinocchio::rnea(model, data, q, v, a);
  forwardKinematics(model,data,q);
  //const pinocchio::SE3 iMd = data.oMi[JOINT_ID].actInv(oMdes);

  // Print out to the vector of joint torques (in N.m)
  std::cout << "q: " << q.transpose() << std::endl;
  std::cout << "v: " << v.transpose() << std::endl;
  std::cout << "a: " << a.transpose() << std::endl;
  std::cout << "Joint torques: " << data.tau.transpose() << std::endl;
  std::cout << "End tran: " << data.oMi[JOINT_ID].translation().transpose()<< std::endl;
  std::cout << "End Pose: " << data.oMi[JOINT_ID].rotation().transpose()<< std::endl;
  return 0;
}
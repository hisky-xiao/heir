// RUN: heir-opt --layout-propagation=min-slot-count=8192 %s | FileCheck %s

#bicyclic = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (65i0 - 66i1 + slot) mod 2145 = 0 and 0 <= i0 <= 32 and 0 <= i1 <= 64 and 0 <= slot <= 8191 }">

// CHECK: #[[REDUCED_LAYOUT:.*]] = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 33 = 0 and 0 <= i0 <= 32 and 0 <= slot <= 8191 }">

// CHECK: func.func @reduce_bicyclic
// CHECK: linalg.reduce
// CHECK-NOT: tensor_ext.convert_layout
// CHECK-SAME: tensor_ext.layout = #[[REDUCED_LAYOUT]]
module {
  func.func @reduce_bicyclic(%arg0: !secret.secret<tensor<33x65xf32>> {tensor_ext.layout = #bicyclic}) -> !secret.secret<tensor<33xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<33xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<33x65xf32>>) {
    ^body(%input0: tensor<33x65xf32>):
      %reduced = linalg.reduce ins(%input0 : tensor<33x65xf32>) outs(%cst : tensor<33xf32>) dimensions = [1]
        (%in: f32, %init: f32) {
          %1 = arith.addf %in, %init : f32
          linalg.yield %1 : f32
        }
      secret.yield %reduced : tensor<33xf32>
    } -> !secret.secret<tensor<33xf32>>
    return %0 : !secret.secret<tensor<33xf32>>
  }
}

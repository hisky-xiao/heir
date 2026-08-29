// RUN: heir-opt --convert-to-ciphertext-semantics %s | FileCheck %s

#kernel = #secret.kernel<name = "BatchMatmulTricyclicDiagonal", force = false>

#layout_out = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (21i0 + 14i1 + 6i2 + slot) mod 42 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 2 and 0 <= i2 <= 6 and 0 <= slot <= 1023 }">
#layout_ct = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (-15i0 - 10i1 - 6i2 + slot) mod 30 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 2 and 0 <= i2 <= 4 and 0 <= slot <= 1023 }">
#layout_pt = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : (35i0 + 14i1 + 20i2 - 14ct + slot) mod 70 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 4 and 0 <= i2 <= 6 and 0 <= ct <= 4 and 0 <= slot <= 1023 }">

#layout_pt2 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : (15i0 - 10i1 - 6i2 - 6ct + slot) mod 30 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 2 and 0 <= i2 <= 4 and 0 <= ct <= 4 and 0 <= slot <= 1023 }">
#layout_ct2 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (-35i0 - 56i1 - 50i2 + slot) mod 70 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 4 and 0 <= i2 <= 6 and 0 <= slot <= 1023 }">

module {
  // CHECK: @batch_matmul_ctpt
  // CHECK-NOT: linalg.batch_matmul
  // CHECK: tensor_ext.rotate
  // CHECK: arith.mulf
  // CHECK: tensor_ext.remap
  func.func @batch_matmul_ctpt(%arg0: !secret.secret<tensor<2x3x5xf32>> {tensor_ext.layout = #layout_ct}, %arg1: tensor<2x5x7xf32>) -> (!secret.secret<tensor<2x3x7xf32>> {tensor_ext.layout = #layout_out}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x3x7xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x3x5xf32>> {tensor_ext.layout = #layout_ct}) {
    ^body(%input0: tensor<2x3x5xf32>):
      %1 = tensor_ext.assign_layout %arg1 {layout = #layout_pt, tensor_ext.layout = #layout_pt} : tensor<2x5x7xf32>
      %2 = tensor_ext.assign_layout %cst {layout = #layout_out, tensor_ext.layout = #layout_out} : tensor<2x3x7xf32>
      %3 = linalg.batch_matmul {secret.kernel = #kernel, tensor_ext.layout = #layout_out, tensor_ext.batch_matmul_period = 6 : i64} ins(%input0, %1 : tensor<2x3x5xf32>, tensor<2x5x7xf32>) outs(%2 : tensor<2x3x7xf32>) -> tensor<2x3x7xf32>
      secret.yield %3 : tensor<2x3x7xf32>
    } -> (!secret.secret<tensor<2x3x7xf32>> {tensor_ext.layout = #layout_out})
    return %0 : !secret.secret<tensor<2x3x7xf32>>
  }

  // CHECK: @batch_matmul_ptct
  // CHECK-NOT: linalg.batch_matmul
  // CHECK: tensor_ext.rotate
  // CHECK: arith.mulf
  // CHECK: tensor_ext.remap
  func.func @batch_matmul_ptct(%arg0: tensor<2x3x5xf32>, %arg1: !secret.secret<tensor<2x5x7xf32>> {tensor_ext.layout = #layout_ct2}) -> (!secret.secret<tensor<2x3x7xf32>> {tensor_ext.layout = #layout_out}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x3x7xf32>
    %0 = secret.generic(%arg1: !secret.secret<tensor<2x5x7xf32>> {tensor_ext.layout = #layout_ct2}) {
    ^body(%input0: tensor<2x5x7xf32>):
      %1 = tensor_ext.assign_layout %arg0 {layout = #layout_pt2, tensor_ext.layout = #layout_pt2} : tensor<2x3x5xf32>
      %2 = tensor_ext.assign_layout %cst {layout = #layout_out, tensor_ext.layout = #layout_out} : tensor<2x3x7xf32>
      %3 = linalg.batch_matmul {secret.kernel = #kernel, tensor_ext.layout = #layout_out, tensor_ext.batch_matmul_period = 14 : i64} ins(%1, %input0 : tensor<2x3x5xf32>, tensor<2x5x7xf32>) outs(%2 : tensor<2x3x7xf32>) -> tensor<2x3x7xf32>
      secret.yield %3 : tensor<2x3x7xf32>
    } -> (!secret.secret<tensor<2x3x7xf32>> {tensor_ext.layout = #layout_out})
    return %0 : !secret.secret<tensor<2x3x7xf32>>
  }
}

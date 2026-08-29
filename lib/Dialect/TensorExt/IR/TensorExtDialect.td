#ifndef LIB_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_
#define LIB_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def TensorExt_Dialect : Dialect {
  let name = "tensor_ext";
  let description = [{
    The `tensor_ext` dialect contains operations on plaintext tensors that
    correspond to the computation model of certain FHE schemes, but are
    unlikely to be upstreamed to MLIR due to their specificity to FHE.
  }];

  let cppNamespace = "::mlir::heir::tensor_ext";
  let dependentDialects = [
    "tensor::TensorDialect",
  ];

  let extraClassDeclaration = [{
    constexpr const static ::llvm::StringLiteral
        kLayoutAttrName = "tensor_ext.layout";
    constexpr const static ::llvm::StringLiteral
        kOriginalTypeAttrName = "tensor_ext.original_type";
    // Carries the resolved ciphertext batch matmul period computed during
    // layout propagation to ciphertext semantics conversion.
    constexpr const static ::llvm::StringLiteral
        kBatchMatmulPeriodAttrName = "tensor_ext.batch_matmul_period";
  }];


  let useDefaultAttributePrinterParser = 1;
}

#endif  // LIB_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/schedule_pass.h>
#include <tvm/schedule.h>
#include "../../src/codegen/codegen_c.h"

TEST(Tensor, Basic) {
    using namespace tvm;
    Var m("m"), n("n");

    Tensor A = Placeholder({m, n}, Float(32), "A");
    auto C = Compute({m, n}, [&](Var i, Var j) {
        return A[i][j];
    }, "C");

    Schedule s = Schedule({C->op});

    auto bounds = schedule::InferBound(s);
    auto stmt = ir::ScheduleOps(s, bounds);

    Buffer Ab = Buffer(A->shape, A->dtype, "A");
    Buffer Cb = Buffer(C->shape, C->dtype, "C");
    Map<Tensor, Buffer> extern_buffer;
    extern_buffer.Set(A, Ab);
    extern_buffer.Set(C, Cb); // If `C` is not set as extern buffer, tvm will allocate for C on stack.
    stmt = ir::StorageFlatten(stmt, extern_buffer);
    LOG(INFO) << "storage flattened stmt:\n" << stmt;

    auto code = codegen::CodeGenC().Compile(stmt, "copy", {Ab->ptr, Cb->ptr, m, n}, false);
    LOG(INFO) << "code:\n" << code;
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}

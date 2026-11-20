import unittest
from copy import deepcopy

import torch
from sru import SRU


class CompileTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_compile_bf16(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 is not supported")
        for bidirectional in (False, True):
            for project_input in (False, True):
                with self.subTest(
                    bidirectional=bidirectional, project_input=project_input
                ):
                    torch.manual_seed(7)
                    input_size = 8 * (2 if bidirectional else 1) + (
                        4 if project_input else 0
                    )
                    eager = SRU(
                        input_size, 8, num_layers=2, bidirectional=bidirectional
                    ).cuda()
                    compiled_layer = deepcopy(eager)
                    x = torch.randn(
                        12, 2, input_size, device="cuda", requires_grad=True
                    )
                    x_compiled = x.detach().clone().requires_grad_()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h, c = eager(x)
                    (h.square().mean() + c.square().mean()).backward()

                    compiled = torch.compile(
                        compiled_layer, backend="inductor", fullgraph=True
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h_compiled, c_compiled = compiled(x_compiled)
                    (h_compiled.square().mean() + c_compiled.square().mean()).backward()

                    torch.testing.assert_close(h_compiled, h, atol=1e-4, rtol=1e-3)
                    torch.testing.assert_close(c_compiled, c, atol=1e-4, rtol=1e-3)
                    torch.testing.assert_close(
                        x_compiled.grad, x.grad, atol=1e-4, rtol=1e-2
                    )
                    for expected, actual in zip(
                        eager.parameters(), compiled_layer.parameters()
                    ):
                        self.assertTrue(torch.isfinite(actual.grad).all())
                        torch.testing.assert_close(
                            actual.grad, expected.grad, atol=1e-4, rtol=1e-2
                        )


if __name__ == "__main__":
    unittest.main()

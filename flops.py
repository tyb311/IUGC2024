from baseline.model import *


if __name__ == '__main__':

    # model = smpdv3()
    model = load_cls_model('res34')
    model.eval()

    inputs=torch.rand(1,3,256,256)
    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, parameter_count_table
    flops = FlopCountAnalysis(model, inputs)
    print(flops)
    print(f"total flops (G): {flops.total()/1e9}")
    
    # acts = ActivationCountAnalysis(model, inputs)
    # print(f"total activations: {acts.total()}")

    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameter(M): {param/1e6}")

    # total flops (G): 55.959103168
    # number of parameter(M): 4.488709

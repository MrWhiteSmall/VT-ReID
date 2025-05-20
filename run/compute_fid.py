import torch_fidelity

fake_path='/public/home/yangzhe/ltt/lsj/OOTD_result'
real_path='/public/home/yangzhe/ltt/lsj/datasets/VITON-HD/test/image'

metrics = torch_fidelity.calculate_metrics(
    input1=fake_path,
    input2=real_path,
    cuda=True, 
    isc=True, 
    fid=True, 
    kid=True, 
    prc=True, 
    verbose=False,
    kid_subset_size=62
)

print(metrics)
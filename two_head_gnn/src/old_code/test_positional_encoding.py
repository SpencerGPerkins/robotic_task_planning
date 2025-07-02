import torch
from pos_encoding import SpatialPositionalEncoding
import matplotlib.pyplot as plt


model = SpatialPositionalEncoding(d_model=60, in_dim=3)
t_0 = torch.tensor([0.32954907417297363, 0.4249360263347626, 0.00251], dtype=torch.float64) * 10
t_1 = torch.tensor([0.32954907417297363, 0.4249360263347626, 0.076], dtype=torch.float64) * 10
# t_0 = torch.tensor([0.3447992205619812,0.4457240402698517, 0.00251], dtype=torch.float64)
# t_1 = torch.tensor([0.32954907417297363, 0.4249360263347626, 0.076], dtype=torch.float64)

pe1 = model(t_0.unsqueeze(0).unsqueeze(0))
pe2 = model(t_1.unsqueeze(0).unsqueeze(0))



l2_diff = torch.norm(pe1 - pe2, p=2).item()
cos_diff = 1 - torch.nn.functional.cosine_similarity(pe1, pe2, dim=-1).item()

print(f"L2 Distance: {l2_diff}")
print(f"Cosine Distance: {cos_diff}")

# Extract vectors from batch
pe1_vec = pe1[0, 0].detach().cpu().numpy()
pe2_vec = pe2[0, 0].detach().cpu().numpy()

# Plot side-by-side line plots
plt.figure(figsize=(12, 5))
plt.plot(pe1_vec, label='Encoding at t0')
plt.plot(pe2_vec, label='Encoding at t1')
plt.title('Positional Encoding Features')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.legend()
plt.show()
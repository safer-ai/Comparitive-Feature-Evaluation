from src.utils import ActivationsDataset
from tqdm import tqdm  # type: ignore
import torch
from src.constants import device
from src.utils import project


def inlp(
    ds: ActivationsDataset,
    n_dim: int = 8,
    n_training_iters: int = 400,
    weight_decay: float = 1e-4,
    learning_rate: float = 1e-4,
) -> torch.Tensor:
    """Compute directions using INLP.

    INLP by Ravfogel, 2020: see https://aclanthology.org/2020.acl-main.647/"""

    working_ds = ActivationsDataset(
        torch.clone(ds.x_data), torch.clone(ds.y_data[:, None].float())
    )
    tot_n_dims = ds.x_data.shape[-1]
    dirs: list[torch.Tensor] = []

    for i in tqdm(range(n_dim)):
        model = torch.nn.Linear(tot_n_dims, 1).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        dataloader = torch.utils.data.DataLoader(
            working_ds, batch_size=256, shuffle=True
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()

        for _ in range(n_training_iters):
            for x, y in dataloader:
                optimizer.zero_grad()
                out = model(x)
                loss_val = loss_fn(out, y)
                loss_val.backward()
                optimizer.step()

        dir = model.weight.detach()[0]

        if dirs:
            dir = project(dir, torch.stack(dirs))
        dir = dir / torch.linalg.norm(dir)

        working_ds.project_(dir)

        dirs.append(dir)
    return torch.stack(dirs)

import torch 
import numpy as np 
import tabnet 
import shutil 
import json 
import zipfile 
import torch.nn.functional as F 
import io 

import torch.nn.functional as F
import pytorch_lightning as pl 
from scipy.sparse import csc_matrix 
from pathlib import Path 
from pytorch_tabnet.utils import (
    create_explain_matrix,
    ComplexEncoder,
)

class TabNetLightning(pl.LightningModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.network = tabnet.tab_network.TabNet(*args, **kwargs)
        self.reducing_matrix = create_explain_matrix(
            self.network.input_dim,
            self.network.cat_emb_dim,
            self.network.cat_idxs,
            self.network.post_embed_dim,
        )

    def forward(self, x):
        return self.base_model.forward(x)

    def _compute_loss(self, y, y_hat):

        # If user doesn't specify, just set to cross_entropy
        if self.loss is None:
            self.loss = F.cross_entropy 

        return self.loss(y, y_hat, weight=self.weights)    

    def _step(self, y, batch):
        output, M_loss = self.network(batch)

        loss = self.compute_loss(output, y)
        # Add the overall sparsity loss
        loss = loss - self.lambda_sparse * M_loss

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass 

    def test_step(self, batch, batch_idx):
        pass 

    def configure_optimizers(self):
        pass 

    def explain(self, loader, normalize=False):
        self.network.eval()
        res_explain = []

        for batch_nb, data in enumerate(loader):
            data = data.to(self.device).float()

            M_explain, masks = self.network.forward_masks(data)
            for key, value in masks.items():
                masks[key] = csc_matrix.dot(
                    value.cpu().detach().numpy(), self.reducing_matrix
                )

            original_feat_explain = csc_matrix.dot(M_explain.cpu().detach().numpy(),
                                                   self.reducing_matrix)
            res_explain.append(original_feat_explain)

            if batch_nb == 0:
                res_masks = masks
            else:
                for key, value in masks.items():
                    res_masks[key] = np.vstack([res_masks[key], value])

        res_explain = np.vstack(res_explain)

        if normalize:
            res_explain /= np.sum(res_explain, axis=1)[:, None]

        return res_explain, res_masks

    def _compute_feature_importances(self, dataloader):
        M_explain, _ = self.explain(dataloader, normalize=False)
        sum_explain = M_explain.sum(axis=0)
        feature_importances_ = sum_explain / np.sum(sum_explain)
        return feature_importances_

    def save_model(self, path):
        saved_params = {}
        init_params = {}
        for key, val in self.get_params().items():
            if isinstance(val, type):
                # Don't save torch specific params
                continue
            else:
                init_params[key] = val
        saved_params["init_params"] = init_params

        class_attrs = {
            "preds_mapper": self.preds_mapper
        }
        saved_params["class_attrs"] = class_attrs

        # Create folder
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save models params
        with open(Path(path).joinpath("model_params.json"), "w", encoding="utf8") as f:
            json.dump(saved_params, f, cls=ComplexEncoder)

        # Save state_dict
        torch.save(self.network.state_dict(), Path(path).joinpath("network.pt"))
        shutil.make_archive(path, "zip", path)
        shutil.rmtree(path)
        print(f"Successfully saved model at {path}.zip")
        return f"{path}.zip"


    def load_model(self, filepath):
        try:
            with zipfile.ZipFile(filepath) as z:
                with z.open("model_params.json") as f:
                    loaded_params = json.load(f)
                    loaded_params["init_params"]["device_name"] = self.device_name
                with z.open("network.pt") as f:
                    try:
                        saved_state_dict = torch.load(f, map_location=self.device)
                    except io.UnsupportedOperation:
                        # In Python <3.7, the returned file object is not seekable (which at least
                        # some versions of PyTorch require) - so we'll try buffering it in to a
                        # BytesIO instead:
                        saved_state_dict = torch.load(
                            io.BytesIO(f.read()),
                            map_location=self.device,
                        )
        except KeyError:
            raise KeyError("Your zip file is missing at least one component")

        self.__init__(**loaded_params["init_params"])

        self._set_network()
        self.network.load_state_dict(saved_state_dict)
        self.network.eval()
        self.load_class_attrs(loaded_params["class_attrs"])

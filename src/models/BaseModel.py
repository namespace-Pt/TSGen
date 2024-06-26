import os
import time
import torch
import psutil
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from typing import Optional, Mapping
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from utils.util import load_pickle, save_pickle, compute_metrics, compute_metrics_nq, makedirs, readlink, synchronize, BaseOutput, MasterLogger, Config
from utils.index import *



class BaseModel(nn.Module):
    """
    Base class for all models.
    """
    def __init__(self, config:Config):
        """
        Args:
            config: the configuration object intialized by :func:`utils.manager.Manager.setup`
            name: the name of the model

        Attributes:
            metrics(dict): the metric dictionary containing ``metric_type: metric_value`` pairs
            config
            name(str): the name of the model
            index_dir(str): the folder to save index e.g. :mod:`utils.index.FaissIndex` and :mod:`utils.index.InvertedVectorIndex`
            collection_dir(str): the folder to save json collections returned by e.g. :func:`utils.index.AnseriniBM25Index.fit`
            encode_dir(str): the folder to save text encoding memmap file returned by :func:`models.BaseModel.BaseSparseModel.encode_text`
            query_dir(str): the folder to save query encoding memmap file returned by :func:`models.BaseModel.BaseSparseModel.encode_query`
            retrieval_result_path(str): the path of the final retrieval result file returned by :func:`models.BaseModel.BaseModel.retrieve`
            _rank(int): the current process ID
            _world_size(int): the number of all processes
            _distributed(bool): if distributed training/evaluating is enabled
            logger(MasterLoger): the logger
        """
        super().__init__()

        # the model's performance, populated when evaluating
        self.metrics = {}
        self.config = config
        self.name = config.name

        # all the following attributes can be generated according to name
        self.retrieve_dir = os.path.join(config.cache_root, config.eval_mode, self.name, config.eval_set)
        self.retrieval_result_path = os.path.join(self.retrieve_dir, str(self.config.save_res) + ".pkl")
        # refered by transformer trainer
        self.ckpt_dir = os.path.join(config.cache_root, "ckpts", self.name)
        self.index_dir = os.path.join(config.cache_root, "index", self.name, config.text_type, config.index_type)
        self.text_dir = os.path.join(config.cache_root, "encode", self.name, "text", config.text_type)
        self.query_dir = os.path.join(config.cache_root, "encode", self.name, "query", config.eval_set)

        self.logger = MasterLogger(self.name)


    def _set_encoder(self, transformer_class=None):
        if transformer_class is None:
            transformer_class = AutoModel

        if self.config.get("untie_encoder"):
            self.queryEncoder = transformer_class.from_pretrained(self.config.plm_dir)
            self.textEncoder = transformer_class.from_pretrained(self.config.plm_dir)

        else:
            plm = transformer_class.from_pretrained(self.config.plm_dir)
            self.queryEncoder = plm
            self.textEncoder = plm

        if hasattr(self.queryEncoder, "pooler"):
            self.queryEncoder.pooler = None
            self.textEncoder.pooler = None
        
        # NOTE: set hidden size to be involked when using deepspeed
        self.config.hidden_size = self.textEncoder.config.hidden_size


    def _move_to_device(self, data, exclude_keys=["text_idx", "query_idx"]):
        """
        Move data to device.

        Args:
            exclude_keys: variables that should be kept unchanged
        """
        if isinstance(data, Mapping):
            new_data = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    if k not in exclude_keys:
                        new_data[k] = v.to(device=self.config.device)
                    else:
                        new_data[k] = v
                elif isinstance(v, Mapping):
                    new_data[k] = self._move_to_device(v)
            new_data = type(data)(new_data)
        elif isinstance(data, torch.Tensor):
            return data.to(device=self.config.device)
        return new_data


    def _l2_distance(self, x1:TENSOR, x2:TENSOR) -> TENSOR:
        """
        Compute l2 similarity.

        Args:
            x1: tensor of [B, D]
            x2: tensor of [B, D]
        """
        ip = torch.matmul(x1, x2.transpose(-1, -2))  # B D
        norm_1 = torch.sum(x1 * x1, dim=-1, keepdim=False).unsqueeze(1).expand(-1, x2.size(0))  # B D
        norm_2 = torch.sum(x2 * x2, dim=-1, keepdim=False).unsqueeze(0).expand(x1.size(0), -1)  # B D
        return norm_1 + norm_2 - 2 * ip


    def _cos_sim(self, x1:TENSOR, x2:TENSOR, temperature:float=0.1) -> TENSOR:
        """
        Compute cosine similarity.

        Args:
            x1: tensor of [B, D]
            x2: tensor of [B, D]
            temperature: scale the similarity scores by dividing temperature
        """
        # x1 = F.normalize(x1, dim=-1)
        # x2 = F.normalize(x2, dim=-1)
        return x1.matmul(x2.transpose(-1,-2)) / temperature


    def _compute_teacher_score(self, x):
        """
        Compute teacher score in knowledge distillation; return None if training in contrastive mode.
        """
        if self.config.enable_distill:
            if "teacher_score" in x:
                teacher_score = x["teacher_score"]  # B, 1+N

            elif "query_teacher_embedding" in x:
                query_teacher_embedding = x["query_teacher_embedding"]  # B, D
                text_teacher_embedding = x["text_teacher_embedding"]    # B, (1+N), D
                if self.config.is_distributed and self.config.enable_all_gather:
                    query_teacher_embedding = self._gather_tensors(query_teacher_embedding)
                    text_teacher_embedding = self._gather_tensors(text_teacher_embedding)
                B, D = query_teacher_embedding.shape
                teacher_score = query_teacher_embedding.matmul(text_teacher_embedding.view(-1, D).transpose(-1,-2)) # B, B * (1 + N)

                if not self.config.enable_inbatch_negative:
                    teacher_score = teacher_score.view(B, B, -1)[range(B), range(B)]    # B, 1 + N

            else:
                raise ValueError("At least teacher_score or query/text teacher embedding should be provided in knowledge distillation!")

        else:
            teacher_score = None
        return teacher_score


    def _compute_loss(self, score:TENSOR, label:TENSOR, teacher_score:Optional[TENSOR]=None):
        """
        A general method to compute loss (contrastive cross-entropy or distillation)

        Args:
            score: tensor of [B, *]
            label: tensor of [B, *]
            x: the input data
        """
        if teacher_score is None:
            loss = F.cross_entropy(score, label)
        else:
            assert score.shape == teacher_score.shape, f"Teacher score {teacher_score.shape} and student score {score.shape} mismatch!"

            label = F.softmax(teacher_score, dim=-1)
            score = F.log_softmax(score, dim=-1)
            loss = torch.mean(-torch.sum(label * score, dim=-1))
        return loss


    def create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer; Subclass may override this function to create custom optimizers. Return None to use the default optimizer created by Trainer.

        Returns:
            optimizer
        """
        return None


    def _gather_objects(self, local_object:object) -> list[object]:
        """
        Gather common python objects across processes.

        .. note::
            This function implicitly consumes GPU.

        Args:
            local_object: python object to collect
        """
        all_objects = [None for _ in range(self.config.world_size)]
        dist.all_gather_object(all_objects, local_object)
        return all_objects


    def _gather_tensors(self, local_tensor:TENSOR) -> TENSOR:
        """
        Gather tensors from all gpus on each process.

        Args:
            local_tensor: the tensor that needs to be gathered

        Returns:
            concatenation of local_tensor in each process
        """
        if local_tensor is None:
            return None
        all_tensors = [torch.empty_like(local_tensor) for _ in range(self.config.world_size)]
        dist.all_gather(all_tensors, local_tensor.contiguous())
        all_tensors[self.config.rank] = local_tensor
        return torch.cat(all_tensors, dim=0)


    def save_to_mmp(self, path:str, shape:tuple, dtype:np.dtype, loader:DataLoader, obj:np.ndarray, batch_size:int=1000):
        """
        #. Create a ``np.memmap`` file of ``shape`` with ``dtype``;

        #. Create lock;

        #. Save the ``obj`` to the offset :attr:`utils.util.Sequential_Sampler.start`;

        #. Release lock.

        Args:
            path: the memmap file path
            shape: the shape of the memmap file to be created
            dtype:
            loader: the dataloader for the data
            obj: the array to be stored
            batch_size: saving in batch
        """
        if self.config.is_main_proc:
            save_dir = os.path.split(path)[0]

            if os.path.exists(path):
                os.remove(path)
            else:
                os.makedirs(save_dir, exist_ok=True)

            lock_path = os.path.join(save_dir, "lock")
            i = 0
            while os.path.exists(lock_path):
                if i == 0:
                    self.logger.info("found lock, waiting for other programs...")
                time.sleep(3)
                i = 1

            save_pickle("this is a lock", lock_path)
            mmp = np.memmap(
                path,
                shape=shape,
                mode="w+",
                dtype=dtype
            )
            del mmp
        # make sure the memmap file has been created
        synchronize()

        self.logger.info(f"saving at {path}")
        mmp = np.memmap(
            path,
            shape=shape,
            mode="r+",
            dtype=dtype
        )

        start_idx = loader.sampler.start
        end_idx = loader.sampler.end
        max_length = shape[0]
        # add in batch
        if max_length > batch_size:
            for i in tqdm(range(start_idx, end_idx, batch_size), leave=False, ncols=100):
                mmp[i: min(i + batch_size, end_idx)] = obj[i - start_idx: i - start_idx + batch_size]
        else:
            mmp[start_idx: end_idx] = obj

        if self.config.is_main_proc:
            # remove the lock
            os.remove(lock_path)


    def gather_retrieval_result(self, retrieval_result:RETRIEVAL_MAPPING, hits: Optional[int]=None, retrieval_result_path: Optional[str]=None) -> RETRIEVAL_MAPPING:
        """
        #. Gather ``retrieval_result`` across processes;

        #. Returning the reordered result cut off to top k;

        #. Create a lock;

        #. Save the result at :attr:`models.BaseModel.BaseModel.retrieval_result_path`.

        #. Release the lock.

        Args:
            retrieval_result: each tuple is a document id-score pair
        Returns:
            processed retrieval result
        """
        if hits is None:
            hits = self.config.hits if self.config.get("verifier_type", "none") == "none" else self.config.verifier_hits
        if retrieval_result_path is None:
            retrieval_result_path = self.retrieval_result_path

        retrieval_result_name = Path(retrieval_result_path).stem

        if self.config.is_main_proc:
            makedirs(retrieval_result_path)

        # create lock for saving and reading the temporary retrieval result
        # check if the lock exists, wait until the lock is released
        lock_path = os.path.join(self.retrieve_dir, f"lock")
        i = 0
        while os.path.exists(lock_path):
            if i == 0:
                self.logger.info("found lock, waiting for other programs...")
            time.sleep(3)
            i += 1

        # make sure every process jump out of the detecting-lock loop before creating a new lock
        synchronize()

        if self.config.is_main_proc:
            save_pickle("this is a lock", lock_path)

        self.logger.info(f"saving retrieval results at {retrieval_result_path}...")

        if self.config.is_distributed:
            local_retrieval_result_path = f"{retrieval_result_path}.{self.config.rank}"
            save_pickle(retrieval_result, local_retrieval_result_path)
            # make sure all processes save the retrieval_result
            synchronize()

            # collect the retrieval result only on master node
            retrieval_result = defaultdict(list)
            if self.config.is_main_proc:
                for i in tqdm(range(self.config.world_size), desc="Merging Retrieval Results", ncols=100, leave=False):
                    tmp_path = f"{retrieval_result_path}.{i}"
                    output = load_pickle(tmp_path)
                    for k, v in output.items():
                        retrieval_result[k].extend(v)
                    os.remove(tmp_path)

        if self.config.is_main_proc:
            # the value of a retrieval_result key is a list of tuple (id, score) or just an id
            try:
                with_score = isinstance(next(iter(retrieval_result.values()))[0], tuple)
            except:
                with_score = False

            if self.config.save_score:
                if not with_score:
                    self.logger.warning("The retrieval result has no score attached, ignoring save_score!")
                retrieval_result_with_scores = defaultdict(list)

            # sort retrieval result
            for qidx, res in retrieval_result.items():
                if hits > 0:
                    if with_score:
                        res = sorted(res, key=lambda x: x[1], reverse=True)
                    reorder_result = res[:hits]
                else:
                    reorder_result = res

                retrieval_result[qidx] = [item[0] if with_score else item for item in reorder_result]

                if self.config.save_score and with_score:
                    retrieval_result_with_scores[qidx] = reorder_result

            # save result
            save_pickle(retrieval_result, retrieval_result_path)
            if self.config.save_score:
                save_pickle(retrieval_result_with_scores, os.path.join(self.retrieve_dir, f"{retrieval_result_name}_with_scores.pkl"))

            # remove the lock
            os.remove(lock_path)

        return retrieval_result


    def init_verifier(self, loaders:LOADERS, load_all_verifier:bool=False):
        """
        Initialize post verifier defined in :pyobj:``utils.index.VERIFIER_MAP``.

        Args:
            loaders
            load_all_verifier: if ``True``, load all the verifier embeddings/codes
        """
        if self.config.get("verifier_type", "none") == "none":
            return None

        else:
            loader_query = loaders["query"]
            loader_text = loaders["text"]

            if load_all_verifier:
                start_text_idx = start_query_idx = 0
                end_text_idx = len(loader_text.dataset)
                end_query_idx = len(loader_query.dataset)
            else:
                start_text_idx = loader_text.sampler.start
                end_text_idx = loader_text.sampler.end
                start_query_idx = loader_query.sampler.start
                end_query_idx = loader_query.sampler.end

            self.logger.info(f"initilizing verifier {self.config.verifier_src}:{self.config.verifier_type}...")

            query_embeddings = np.memmap(
                # the embedding file may be a symbolic link
                readlink(os.path.join(self.config.cache_root, "encode", self.config.verifier_src, "query", self.config.eval_set, "query_embeddings.mmp")),
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_query.dataset), -1)[start_query_idx: end_query_idx].copy()

            text_embeddings = pq_index = None
            if self.config.verifier_type == "flat":
                text_embeddings = np.memmap(
                    # the embedding file may be a symbolic link
                    readlink(os.path.join(self.config.cache_root, "encode", self.config.verifier_src, "text", self.config.text_type, "text_embeddings.mmp")),
                    mode="r",
                    dtype=np.float32
                ).reshape(len(loader_text.dataset), -1)[start_text_idx: end_text_idx].copy()
            elif self.config.verifier_type == "pq":
                pq_index = faiss.read_index(os.path.join(self.config.cache_root, "index", self.config.verifier_src, "faiss", self.config.verifier_index))

            verifier = VERIFIER_MAP[self.config.verifier_type](
                query_embeddings=query_embeddings,
                text_embeddings=text_embeddings,
                pq_index=pq_index,
                hits=self.config.verifier_hits,
                device=self.config.device,
                start_text_idx=start_text_idx,
                end_text_idx=end_text_idx,
            )
            return verifier


    def encode(self, loaders):
        """
        Shotcut for encoding both text and query.
        """
        if self.config.do_text:
            self.encode_text(loaders["text"])
        if self.config.do_query:
            self.encode_query(loaders["query"])


    def index(self, loaders:LOADERS):
        """
        The index method. Subclass should override this function.
        """
        pass


    def retrieve(self, loaders:LOADERS):
        """
        The retrieve method. Subclass should override this function.
        """
        pass


    @synchronize
    def rerank(self, loaders: dict):
        """
        Rerank the candidates in :mod:`utils.dataset.PairDataset`.
        """
        loader_rerank = loaders["rerank"]
        retrieval_result = defaultdict(list)

        self.logger.info("reranking...")
        for i, x in enumerate(tqdm(loader_rerank, ncols=100, leave=False)):
            query_idx = x["query_idx"].tolist()	# B
            text_idx = x["text_idx"].tolist()	# B
            score = self.rerank_step(x).cpu().tolist()	# B
            for j, qidx in enumerate(query_idx):
                retrieval_result[qidx].append((text_idx[j], score[j]))

            if self.config.get("debug") and i > 5:
                break
        return retrieval_result


    @synchronize
    @torch.no_grad()
    def evaluate(self, loaders:LOADERS, log:bool=True):
        """
        #. Evaluate the model on ``config.eval_set``;

        #. Log the metrics;

        #. Save the checkpoint if necessary.

        Args:
            log: call `log_result()`?
        """
        self.eval()

        if self.config.load_result:
            retrieval_result = load_pickle(self.retrieval_result_path)

        else:
            if self.config.eval_mode == "rerank":
                retrieval_result = self.rerank(loaders)
            elif self.config.eval_mode == "retrieve":
                # all models should override the retrieve method
                retrieval_result = self.retrieve(loaders)
            else:
                raise ValueError(f"Unsupported Mode {self.config.mode}!")

            # merge retrieval result from all processes if _distributed
            # save retrieval result
            retrieval_result = self.gather_retrieval_result(retrieval_result)

        if self.config.is_main_proc:
            self.logger.info("evaluating...")

            if self.config.dataset == "NQ-open":
                if self.config.eval_set == "train":
                    ground_truth_path = os.path.join(self.config.cache_root, "dataset", "query", self.config.eval_set, "positives.pkl")
                    ground_truth = load_pickle(ground_truth_path)
                    metrics = compute_metrics(retrieval_result, ground_truth, cutoffs=self.config.eval_metric_cutoff)
                elif self.config.eval_set == "dev":
                    metrics = compute_metrics_nq(retrieval_result, os.path.join(self.config.data_root, self.config.dataset, "nq-test.qa.csv"), os.path.join(self.config.data_root, self.config.dataset, "collection.tsv"))
                else:
                    raise NotImplementedError
            else:
                ground_truth_path = os.path.join(self.config.cache_root, "dataset", "query", self.config.eval_set, "positives.pkl")
                ground_truth = load_pickle(ground_truth_path)
                metrics = compute_metrics(retrieval_result, ground_truth, metrics=self.config.eval_metric, cutoffs=self.config.eval_metric_cutoff)

            self.metrics.update(metrics)
            # the model will add some metrics such as FLOPs
            all_metrics = {k: v for k, v in self.metrics.items() if k != "_best"}
            self.logger.info(f"{self.name}: {all_metrics}")

            if log:
                self.log_result()
        
        if self.config.is_distributed:
            if self.config.is_main_proc:
                objects = [all_metrics]
            else:
                # assign none to other processes
                objects = [None]

            # broadcast the metrics to all processes
            dist.broadcast_object_list(objects, src=0)
            all_metrics = objects[0]

        return all_metrics


    def log_result(self, **kwargs):
        """
            Save the model metrics and configurations in ``performance.log``.
        """
        name = self.name
        metrics = {k: v for k, v in self.metrics.items() if k != "_best"}
        metrics.update(kwargs)

        with open("performance.log", "a+") as f:
            d = self.config
            line = "{} : {}\n{}\n".format(name, str(d), str(metrics))
            f.write(line)

            try:
                if self.config.dataset == "NQ-open":
                    markdown_format_metric = "|".join([str(metrics["Recall@5"]), str(metrics["Recall@10"]), str(metrics["Recall@20"]), str(metrics["Recall@100"])]) + "|"
                elif "NQ320k" in self.config.dataset or "MS300k" in self.config.dataset or "MS600k" in self.config.dataset:
                    markdown_format_metric = "|".join([str(metrics["MRR@5"]), str(metrics["MRR@10"]), str(metrics["Recall@5"]), str(metrics["Recall@10"])]) + "|"
                    markdown_format_metric += "\t" + "|".join([str(metrics["MRR@10"]), str(metrics["MRR@100"]), str(metrics["Recall@1"]), str(metrics["Recall@10"]), str(metrics["Recall@100"])]) + "|"
                else:
                    markdown_format_metric = "|".join([str(metrics["MRR@10"]), str(metrics["Recall@10"]), str(metrics["Recall@100"]), str(metrics["Recall@1000"])]) + "|"
            except:
                markdown_format_metric = ""
            if "FLOPs" in metrics:
                markdown_format_metric += str(metrics["FLOPs"]) + "|"
            if "Posting_List_Length" in metrics:
                markdown_format_metric += str(metrics["Posting_List_Length"]) + "|"
            if "X Posting_List_Length" in metrics and "Y Posting_List_Length" in metrics:
                markdown_format_metric += str(metrics["X Posting_List_Length"] + metrics["Y Posting_List_Length"]) + "|"
            f.write(markdown_format_metric + "\n")
            f.write("\n")


    # no synchronize here because maybe we only call this function on main process
    def save(self, checkpoint:Optional[Union[str,int]]=None):
        """
        Save the model at ``checkpoint``.
        """
        # set to eval mode when saving
        if self.training:
            self.eval()
            training = True
        else:
            training = False

        if checkpoint is None:
            checkpoint = self.config.save_ckpt

        save_path = f"{self.config.cache_root}/ckpts/{self.name}/{checkpoint}"

        os.makedirs(os.path.split(save_path)[0], exist_ok=True)

        self.logger.info("saving model at {}...".format(save_path))
        model_dict = self.state_dict()

        if self.config.is_main_proc:
            save_dict = {}
            # the distributed infomation will not be saved
            save_dict["config"] = self.config
            save_dict["model"] = model_dict
            save_dict["metrics"] = {k: v for k, v in self.metrics.items() if k != "_best"}
            torch.save(save_dict, save_path)

        if training:
            self.train()


    def load(self):
        """
        Load the model with current config from ``config.load_ckpt``.
        """
        checkpoint = self.config.load_ckpt

        if checkpoint == "none" or checkpoint is None:
            return
        elif os.path.isfile(str(checkpoint)):
            save_path = checkpoint
        elif os.path.isfile(f"{self.config.cache_root}/ckpts/{checkpoint}"):
            save_path = f"{self.config.cache_root}/ckpts/{checkpoint}"
        else:
            save_path = f"{self.config.cache_root}/ckpts/{self.name}/{checkpoint}"

        if not os.path.exists(save_path):
            self.logger.warning(f"Checkpoint {checkpoint} not found, not loading any checkpoints!")
            return

        self.logger.info("loading model from {}...".format(save_path))

        state_dict = torch.load(save_path, map_location=torch.device(self.config.device))
        missing_keys, unexpected_keys = self.load_state_dict(state_dict["model"], strict=False)

        current_config = self.config
        for k, v in state_dict["config"].items():
            try:
                if v != current_config[k]:
                    self.logger.info(f"model config {k} of the checkpoint is {v}, while it's {current_config[k]} in current config!")
            except KeyError:
                pass

        if len(missing_keys):
            self.logger.warning(f"Missing Keys: {missing_keys}")
        if len(unexpected_keys):
            self.logger.warning(f"Unexpected Keys: {unexpected_keys}")


    def step_end_callback(self, loaders, state):
        """
        Callback at the end of each training step.
        """
        pass



class BaseSparseModel(BaseModel):
    """
    Base class for all models that rely on token weights to rank documents.
    """
    def __init__(self, config:Config):
        super().__init__(config)

        # set to false to include special tokens into the inverted index
        self._skip_special_tokens = True
        # set to 1 by default, meaning the token weight is stored
        self._output_dim = 1
        # posting list number, may be extended for latent topics
        self._posting_entry_num = self.config.vocab_size
        # valid text length for indexing and searching
        self._text_length = self.config.text_length
        self._query_length = self.config.query_length

        # override index_dir
        if self.config.index_type == "impact":
            self.index_dir = os.path.join(self.index_dir, self.config.granularity)
        elif self.config.index_type == "bm25":
            if self.config.pretokenize:
                self.index_dir = os.path.join(self.index_dir, self.config.granularity)
            else:
                self.index_dir = os.path.join(self.index_dir, "default")


    def _compute_overlap(self, query_token_id:TENSOR, text_token_id:TENSOR) -> TENSOR:
        """
        Compute overlapping mask between the query tokens and positive sequence tokens across batches.

        Args:
            query_token_id: [B1, LQ]
            text_token_id: [B2, LS]

        Returns:
            overlapping_mask: [B, LQ, B, LS] if cross_batch, else [B, LQ, LS]
        """
        query_token_id = query_token_id[..., None, None] # B, LQ, 1, 1
        text_token_id = text_token_id[None, None, ...]   # 1, 1, B, LS

        overlapping_mask = text_token_id == query_token_id
        return overlapping_mask


    def _gate_text(self, text_token_weights:np.ndarray, k:Optional[int]=None):
        """
        Gate the text token weights so that only the top ``config.query_gate_k`` tokens are valid. Keep the text_token_ids because we will use it to construct the entire inverted lists.

        Args:
            query_embeddings: [N, L, 1]
        """
        if k is None:
            k = self.config.text_gate_k
        if k > 0 and k < text_token_weights.shape[1]:
            # the original one is read-only
            text_token_weights = text_token_weights.copy()

            self.logger.info(f"gating text by {k}...")
            assert text_token_weights.shape[-1] == 1
            text_token_weights = text_token_weights.squeeze(-1)
            non_topk_indices = np.argpartition(-text_token_weights, k)[:, k:]
            np.put_along_axis(text_token_weights, non_topk_indices, values=0, axis=-1)
            # append the last dimension
            text_token_weights = np.expand_dims(text_token_weights, axis=-1)
        return text_token_weights


    def encode_text_step(self, x):
        """
        One step in encode_text.

        Args:
            x: a data record.

        Returns:
            the text token id for indexing, array of [B, L]
            the text token embedding for indexing, array of [B, L, D]
        """
        text_token_id = x["text"]["input_ids"].numpy()
        text_token_embedding = np.ones((*text_token_id.shape, self._output_dim), dtype=np.float32)

        if "text_first_mask" in x:
            # mask the duplicated tokens' weight
            text_first_mask = x["text_first_mask"].numpy()
            text_token_embedding[~text_first_mask] = 0
        else:
            text_token_embedding[~x["text"]["attention_mask"].bool().numpy()] = 0

        return text_token_id, text_token_embedding


    def encode_query_step(self, x):
        """
        One step in encode_text.

        Args:
            x: a data record.

        Returns:
            the query token id for searching, array of [B, L]
            the query token embedding for indexing, array of [B, L, D]
        """
        query_token_id = x["query"]["input_ids"].numpy()
        query_token_embedding = np.ones((*query_token_id.shape, self._output_dim), dtype=np.float32)
        query_token_embedding[~x["query"]["attention_mask"].bool().numpy()] = 0
        return query_token_id, query_token_embedding


    @synchronize
    @torch.no_grad()
    def encode_text(self, loader_text:DataLoader, load_all_encode:bool=False):
        """
        Encode texts into token weights or token vecs.

        Args:
            load_all_encode: bool, set to true to load the entire cache file

        Returns:
            BaseOutput:
                text_embeddings: array of [N, L, D]
                text_token_ids: array of [N, L]
        """
        text_token_id_path = os.path.join(self.text_dir, "text_token_ids.mmp")
        text_embedding_path = os.path.join(self.text_dir, "text_embeddings.mmp")

        if load_all_encode:
            text_embeddings = np.memmap(
                text_embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_text.dataset), self._text_length, self._output_dim)
            text_token_ids = np.memmap(
                text_token_id_path,
                mode="r",
                dtype=np.int32
            ).reshape(len(loader_text.dataset), self._text_length)

        elif self.config.load_encode or self.config.load_text_encode:
            text_embeddings = np.memmap(
                text_embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_text.dataset), self._text_length, self._output_dim)[loader_text.sampler.start: loader_text.sampler.end]
            text_token_ids = np.memmap(
                text_token_id_path,
                mode="r",
                dtype=np.int32
            ).reshape(len(loader_text.dataset), self._text_length)[loader_text.sampler.start: loader_text.sampler.end]

        else:
            text_token_ids = np.zeros((len(loader_text.sampler), self._text_length), dtype=np.int32)
            text_embeddings = np.zeros((len(loader_text.sampler), self._text_length, self._output_dim), dtype=np.float32)

            start_idx = end_idx = 0
            self.logger.info(f"encoding {self.config.dataset} text...")
            for i, x in enumerate(tqdm(loader_text, leave=False, ncols=100)):
                text_token_id, text_embedding = self.encode_text_step(x)

                end_idx += text_embedding.shape[0]
                text_token_ids[start_idx: end_idx] = text_token_id
                text_embeddings[start_idx: end_idx] = text_embedding
                start_idx = end_idx
                if self.config.debug:
                    if i > 10:
                        break

            if self.config.save_encode:
                self.save_to_mmp(
                    path=text_token_id_path,
                    shape=(len(loader_text.dataset), self._text_length),
                    dtype=np.int32,
                    loader=loader_text,
                    obj=text_token_ids
                )
                self.save_to_mmp(
                    path=text_embedding_path,
                    shape=(len(loader_text.dataset), self._text_length, self._output_dim),
                    dtype=np.float32,
                    loader=loader_text,
                    obj=text_embeddings
                )
        text_embeddings = self._gate_text(text_embeddings)
        return BaseOutput(embeddings=text_embeddings, token_ids=text_token_ids)


    @synchronize
    @torch.no_grad()
    def encode_query(self, loader_query:DataLoader, load_all_encode:bool=False):
        """
        Encode each query into a weight or a vector.

        Args:
            load_all_encode: if ``True``, load all cached memmap from :attr:`models.BaseModel.BaseModel.encode_dir`

        Returns:
            BaseOutput:
                embeddings: np.ndarray of [N, L, D]
                token_ids: np.ndarray of [N, L]
        """
        query_token_id_path = os.path.join(self.query_dir, "query_token_ids.mmp")
        query_embedding_path = os.path.join(self.query_dir, "query_embeddings.mmp")

        if load_all_encode:
            query_embeddings = np.memmap(
                query_embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_query.dataset), self._query_length, self._output_dim)
            query_token_ids = np.memmap(
                query_token_id_path,
                mode="r",
                dtype=np.int32
            ).reshape(len(loader_query.dataset), self._query_length)
        elif self.config.load_encode or self.config.load_query_encode:
            query_embeddings = np.memmap(
                query_embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_query.dataset), self._query_length, self._output_dim)[loader_query.sampler.start: loader_query.sampler.end]
            query_token_ids = np.memmap(
                query_token_id_path,
                mode="r",
                dtype=np.int32
            ).reshape(len(loader_query.dataset), self._query_length)[loader_query.sampler.start: loader_query.sampler.end]
        else:
            query_token_ids = np.zeros((len(loader_query.sampler), self._query_length), dtype=np.int32)
            query_embeddings = np.zeros((len(loader_query.sampler), self._query_length, self._output_dim), dtype=np.float32)

            start_idx = end_idx = 0
            self.logger.info(f"encoding {self.config.dataset} {self.config.eval_set} query...")
            for i, x in enumerate(tqdm(loader_query, leave=False, ncols=100)):
                query_token_id, query_embedding = self.encode_query_step(x)

                end_idx += query_embedding.shape[0]
                query_token_ids[start_idx: end_idx] = query_token_id
                query_embeddings[start_idx: end_idx] = query_embedding
                start_idx = end_idx
                if self.config.debug:
                    if i > 10:
                        break

            if self.config.save_encode:
                self.save_to_mmp(
                    path=query_token_id_path,
                    shape=(len(loader_query.dataset), self._query_length),
                    dtype=np.int32,
                    loader=loader_query,
                    obj=query_token_ids
                )
                self.save_to_mmp(
                    path=query_embedding_path,
                    shape=(len(loader_query.dataset), self._query_length, self._output_dim),
                    dtype=np.float32,
                    loader=loader_query,
                    obj=query_embeddings
                )

        return BaseOutput(embeddings=query_embeddings, token_ids=query_token_ids)


    def inverted_index(self, loader_text:DataLoader):
        """
        Construct :class:`utils.index.BaseInvertedIndex`.
        """
        encode_output = self.encode_text(loader_text)

        text_embeddings = encode_output.embeddings
        text_token_ids = encode_output.token_ids
        text_embeddings_tensor = torch.as_tensor(text_embeddings.copy(), device=self.config.device)

        # invvec and invhit share the same inverted index
        save_dir = os.path.join(self.config.cache_root, "index", self.name, self.config.text_type, "inv", "_".join([self.config.plm_tokenizer, str(self._text_length), ",".join([str(x) for x in self.config.text_col])]), str(self.config.world_size))

        special_token_ids = set()
        if self._skip_special_tokens:
            special_token_ids.update([x[1] for x in self.config.special_token_ids.values() if x[0] is not None])

        index = INVERTED_INDEX_MAP[self.config.index_type](
            text_num=text_embeddings_tensor.shape[0],
            token_num=self._posting_entry_num,
            device=self.config.device,
            rank=self.config.rank,
            save_dir=save_dir,
            special_token_ids=special_token_ids
        )
        index.fit(
            text_token_ids=text_token_ids,
            text_embeddings=text_embeddings_tensor,
            load_index=self.config.load_index,
            save_index=self.config.save_index,
            threads=self.config.get("index_thread", 16) // self.config.world_size,
            shards=self.config.get("index_shard", 32) // self.config.world_size,
            posting_prune=self.config.get("posting_prune", 0),
            start_text_idx=loader_text.sampler.start,
        )

        if self.config.eval_flops:
            return BaseOutput(
                embeddings=text_embeddings,
                token_ids=text_token_ids,
                index=index
            )
        else:
            return BaseOutput(index=index)


    def anserini_index(self, loader_text:DataLoader):
        """
        Construct :class:`utils.index.BaseAnseriniIndex`.
        """
        save_encode = self.config.save_encode
        if not (self.config.load_encode or self.config.load_text_encode):
            self.config.save_encode = True
            self.logger.warning("Automatically set save_encode=True to save encode results on disk for later usage by anserini!")
        encode_output = self.encode_text(loader_text)
        self.config.save_encode = save_encode
    
        # load cache only on the master node
        all_encode_output = self.encode_text(loader_text, load_all_encode=True)
        if not self.config.is_main_proc:
            all_encode_output = None

        if self.config.is_main_proc:
            all_text_token_ids = all_encode_output.token_ids
            all_text_token_weights = all_encode_output.embeddings.squeeze(-1) if all_encode_output.embeddings is not None else None

            if self.config.text_type == "default":
                collection_path = os.path.join(self.config.data_root, self.config.dataset, "collection.tsv")
            else:
                raise NotImplementedError(f"Anserini index for text type {self.config.text_type} is not implemented yet!")
            
            # include plm special tokens
            stop_words = set(x[0] for x in self.config.special_token_ids.values() if x[0] is not None)

            collection_dir = os.path.join(self.index_dir, "collection")
            index_dir = os.path.join(self.index_dir, "index")

            index = ANSERINI_INDEX_MAP[self.config.index_type](
                collection_dir=collection_dir,
                index_dir=index_dir
            )

            if self.config.load_collection:
                enable_build_collection = False
            else:
                enable_build_collection = True

            if self.config.load_index:
                # if load index, then load the collection as well
                enable_build_index = False
                enable_build_collection = False
            else:
                enable_build_index = True

            if self.config.index_type == "impact" and self.config.granularity == "word":
                subword_to_word = SUBWORD_TO_WORD_FN[self.config.plm_tokenizer]
            else:
                subword_to_word = None

            index.fit(
                text_path=collection_path,
                text_cols=self.config.text_col,
                text_token_ids=all_text_token_ids,
                text_token_weights=all_text_token_weights,
                tokenizer=AutoTokenizer.from_pretrained(self.config.plm_dir),
                stop_words=stop_words,
                thread_num=self.config.index_thread,
                enable_build_collection=enable_build_collection,
                enable_build_index=enable_build_index,
                language=self.config.language,
                granularity=self.config.granularity,
                # sepecific to impact indexes
                subword_to_word=subword_to_word,
                quantize_bit=self.config.get("quantize_bit"),
                reduce=self.config.get("reduce"),
            )

        else:
            index = None

        if self.config.eval_flops:
            return BaseOutput(
                token_ids=encode_output.token_ids,
                embeddings=encode_output.embeddings,
                index=index
            )
        else:
            return BaseOutput(index=index)


    @synchronize
    def index(self, loaders:LOADERS):
        """
        Wrapper to construct a variety of sparse indexes. Subclass may override this function to create customized index.
        """
        if self.config.index_type in INVERTED_INDEX_MAP:
            return self.inverted_index(loaders["text"])
        elif self.config.index_type in ANSERINI_INDEX_MAP:
            return self.anserini_index(loaders["text"])
        else:
            raise NotImplementedError


    @synchronize
    def retrieve(self, loaders:LOADERS) -> RETRIEVAL_MAPPING:
        """
        #. Retrieve by the index;

        #. Compute auxillary metrics if necessary;

        #. Post verify if necessary.

        Returns:
            retrieval result
        """
        loader_query = loaders["query"]

        output = self.index(loaders)
        index = output.index

        encode_output = self.encode_query(loader_query)
        query_embeddings = encode_output.embeddings
        query_token_ids = encode_output.token_ids

        # use anserini to retrieve
        if isinstance(index, BaseAnseriniIndex):
            # anserini index only on the main process
            os.makedirs(self.retrieve_dir, exist_ok=True)

            tid2index = load_pickle(os.path.join(self.config.cache_root, "dataset", "text", "id2index.pkl"))
            qid2index = load_pickle(os.path.join(self.config.cache_root, "dataset", "query", self.config.eval_set, "id2index.pkl"))

            query_path = f"{self.config.data_root}/{self.config.dataset}/queries.{self.config.eval_set}.tsv"

            # load all verifier embeddings on the master node
            verifier = self.init_verifier(loaders, load_all_verifier=True)

            self.logger.info("searching...")
            retrieval_result = index.search(
                query_token_ids=query_token_ids,
                query_token_weights=query_embeddings.squeeze(-1) if query_embeddings is not None else None,
                query_path=query_path,
                tmp_query_dir=Path(index.index_dir).parent / "query",
                retrieval_result_path=self.retrieval_result_path,
                hits=self.config.hits,
                qid2index=qid2index,
                tid2index=tid2index,
                language=self.config.language,
                k1=self.config.get("k1"),
                b=self.config.get("b"),
                verifier=verifier
            )

        # inverted index and None
        elif isinstance(index, BaseInvertedIndex):
            verifier = self.init_verifier(loaders)

            self.logger.info("searching...")
            retrieval_result, posting_list_length = index.search(
                query_token_ids=query_token_ids,
                query_embeddings=query_embeddings,
                # this is useful when performing query side parallel
                query_start_idx=loader_query.sampler.start,
                hits=self.config.hits,
                eval_posting_length=self.config.eval_posting_length,
                verifier=verifier
            )

            # manually delete the index
            del index

            if self.config.eval_posting_length:
                if self.config.is_distributed:
                    posting_list_length = np.asarray(self._gather_objects(posting_list_length.mean())).sum()
                else:
                    posting_list_length = posting_list_length.mean()

                self.metrics["Posting_List_Length"] = int(np.round(posting_list_length))
                self.logger.info(f"Average Posting Length is {self.metrics['Posting_List_Length']}!")

        # non-main process in anserini index
        elif index is None:
            retrieval_result = {}

        else:
            raise NotImplementedError

        if self.config.eval_flops:
            self.compute_flops(loaders, output.token_ids, output.embeddings, query_token_ids, query_embeddings, log=False)

        return retrieval_result


    @torch.no_grad()
    @synchronize
    def compute_flops(self, loaders:LOADERS, text_token_ids:np.ndarray, text_token_weights:np.ndarray, query_token_ids:np.ndarray, query_token_weights:np.ndarray, log:bool=True):
        """
        Compute flops as stated in `SPLADE <https://arxiv.org/pdf/2109.10086.pdf>`_;

        .. note::
            This function uses the cached embedding to compute flops.
        """
        assert self._output_dim == 1
        loader_text = loaders["text"]
        loader_query = loaders["query"]

        self.logger.info("computing flops...")

        text_token_weights = text_token_weights.squeeze(-1)
        if query_token_weights is not None:
            query_token_weights = query_token_weights.squeeze(-1)
        else:
            query_token_weights = np.ones(query_token_ids.shape, dtype=np.float32)

        D = np.zeros(self._posting_entry_num)
        Q = np.zeros(self._posting_entry_num)

        for i, text_token_id in enumerate(tqdm(text_token_ids, ncols=100, desc="Collecting Tokens in Text", leave=False)):
            # ignore the token id whose weight is 0
            text_token_id = text_token_id[text_token_weights[i] != 0]
            D[text_token_id] += 1

        for i, query_token_id in enumerate(tqdm(query_token_ids, ncols=100, desc="Collecting Tokens in Query", leave=False)):
            # ignore the token id whose weight is 0
            query_token_id = query_token_id[query_token_weights[i] != 0]
            Q[query_token_id] += 1

        if self._skip_special_tokens:
            special_token_ids = [x[1] for x in self.config.special_token_ids.values() if x[0] is not None]
            D[special_token_ids] = 0
            Q[special_token_ids] = 0

        D /= len(loader_text.sampler)
        Q /= len(loader_query.sampler)
        flops = Q @ D

        # when distributed, compute flops of each shard and merge by average
        if self.config.is_distributed:
            all_flops = self._gather_objects(flops)
            flops = np.asarray(all_flops).mean()

        flops = round(flops, 2)

        self.metrics.update({"FLOPs": flops})
        if log:
            self.log_result()
            self.logger.info(f"FLOPs: {flops}")


    @torch.no_grad()
    def generate_code(self, loaders:LOADERS):
        """
        Generate codes from the cache embedding files.
        """
        if self.config.is_main_proc:
            from utils.util import _get_token_code

            # the code is bind to the code_tokenizer
            code_path = os.path.join(self.config.cache_root, "codes", self.config.code_type, self.config.code_tokenizer, str(self.config.code_length), "codes.mmp")
            self.logger.info(f"generating codes from {self.config.code_type} with code_length: {self.config.code_length}, saving at {code_path}...")

            loader_text = loaders["text"]
            text_num = len(loader_text.dataset)
            makedirs(code_path)

            # load all saved token ids
            # all codes are led by 0 and padded by -1
            text_codes = np.memmap(
                code_path,
                dtype=np.int32,
                mode="w+",
                shape=(text_num, self.config.code_length)
            )
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.config.plm_root, self.config.code_tokenizer))
            model = AutoModel.from_pretrained(os.path.join(self.config.plm_root, self.config.code_tokenizer))
            try:
                start_token_id = model._get_decoder_start_token_id()
            except ValueError:
                start_token_id = model.config.pad_token_id
                self.logger.warning(f"Decoder start token id not found, use pad token id ({start_token_id}) instead!")
            
            if self.config.get("store_weight"):
                weight_path = os.path.join(self.config.cache_root, "codes", self.config.code_type, self.config.code_tokenizer, str(self.config.code_length), "weights.mmp")
                # store the weights of each semantic unit (elements between the code_sep)
                text_code_weights = np.memmap(
                    weight_path,
                    dtype=np.float32,
                    mode="w+",
                    shape=(text_num, self.config.code_length)
                )
            else:
                weight_path = None

            # the codes are always led by start_token_id and padded by -1
            text_codes[:, 0] = start_token_id
            text_codes[:, 1:] = -1

            code_fields = self.config.code_type.split("-")
            defaults = ["weight", None]
            code_fields.extend(defaults[-(3 - len(code_fields)):])
            code_name, code_init_order, code_post_order = code_fields[:3]

            stop_words = set()
            punctuations = set([x for x in ";:'\\\"`~[]<>()\{\}/|?!@$#%^&*…-_=+,."])
            nltk_stop_words = set(["a", "s", "about", "also", "am", "to", "an", "and", "another", "any", "anyone", "are", "aren't", "as", "at", "be", "been", "being", "but", "by", "despite", "did", "didn't", "do", "does", "doesn't", "doing", "done", "don't", "each", "etc", "every", "everyone", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "her", "here", "here's", "hers", "herself", "he's", "him", "himself", "his", "however", "i", "i'd", "if", "i'll", "i'm", "in", "into", "is", "isn't", "it", "its", "it's", "itself", "i've", "just", "let's", "like", "lot", "may", "me", "might", "mightn't", "my", "myself", "no", "nor", "not", "of", "on", "onto", "or", "other", "ought", "oughtn't", "our", "ours", "ourselves", "out", "over", "shall", "shan't", "she", "she'd", "she'll", "she's", "since", "so", "some", "something", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "tht", "to", "too", "usually", "very", "via", "was", "wasn't", "we", "we'd", "well", "we'll", "were", "we're", "weren't", "we've", "will", "with", "without", "won't", "would", "wouldn't", "yes", "yet", "you", "you'd", "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've"])
            # include punctuations
            stop_words = stop_words | punctuations
            # include nltk stop words
            stop_words = stop_words | nltk_stop_words
            # include numbers in stopwords
            # stop_words.add(r"\d")

            thread_num = 0
            collection_dir = os.path.join(self.index_dir, "collection")
            for path in os.listdir(collection_dir):
                # check if current path is a file
                if os.path.isfile(os.path.join(collection_dir, path)):
                    thread_num += 1

            # each thread creates one jsonl file
            text_num_per_thread = text_num / thread_num

            arguments = []
            # re-tokenize words in the collection folder
            for i in range(thread_num):
                input_path = os.path.join(collection_dir, "docs{:02d}.json".format(i))
                start_idx = round(text_num_per_thread * i)
                end_idx = round(text_num_per_thread * (i+1))

                arguments.append((
                    input_path,
                    code_path,
                    text_num,
                    start_idx,
                    end_idx,
                    tokenizer,
                    self.config.code_length,
                    code_init_order,
                    code_post_order,
                    stop_words,
                    self.config.get("code_sep", " "),
                    self.config.get("stem_code"),
                    self.config.get("filter_num"),
                    self.config.get("filter_unit"),
                    self.config.get("ngram", 1),
                    weight_path
                ))

            # the collection has no special_tokens so we don't need to filter them out
            with mp.Pool(thread_num) as p:
                p.starmap(_get_token_code, arguments)



class BaseDenseModel(BaseModel):
    """
    Base class for all models that rely on sequence embeddings to rank documents.
    """
    def __init__(self, config):
        super().__init__(config)
        # TODO: other ANN libraries
        self.index_dir = os.path.join(self.index_dir, "faiss")


    def encode_text_step(self, x):
        text = self._move_to_device(x["text"])
        embedding = self.textEncoder(**text)[0][:, 0]

        if self.config.dense_metric == "cos":
            embedding = F.normalize(embedding, dim=-1)
        return embedding.cpu().numpy()


    def encode_query_step(self, x):
        query = self._move_to_device(x["query"])
        embedding = self.queryEncoder(**query)[0][:, 0]

        if self.config.dense_metric == "cos":
            embedding = F.normalize(embedding, dim=-1)
        return embedding.cpu().numpy()


    @synchronize
    @torch.no_grad()
    def encode_text(self, loader_text:DataLoader, load_all_encode:bool=False):
        """
        Encode each text into a vector.

        Args:
            load_all_encode: bool, set to true to load the entire cache file

        Returns:
            BaseOutput:
                text_embeddings: array of [N, D]
        """
        text_embedding_path = os.path.join(self.text_dir, "text_embeddings.mmp")

        if load_all_encode:
            text_embeddings = np.memmap(
                text_embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_text.dataset), self._output_dim)

        elif self.config.load_encode or self.config.load_text_encode:
            text_embeddings = np.memmap(
                text_embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_text.dataset), self._output_dim)[loader_text.sampler.start: loader_text.sampler.end]

        else:
            text_embeddings = np.zeros((len(loader_text.sampler), self._output_dim), dtype=np.float32)
            start_idx = end_idx = 0
            self.logger.info(f"encoding {self.config.dataset} text...")
            for i, x in enumerate(tqdm(loader_text, leave=False, ncols=100)):
                text_embedding = self.encode_text_step(x)

                end_idx += text_embedding.shape[0]
                text_embeddings[start_idx: end_idx] = text_embedding
                start_idx = end_idx
                if self.config.debug:
                    if i > 10:
                        break

            if self.config.save_encode:
                self.save_to_mmp(
                    path=text_embedding_path,
                    shape=(len(loader_text.dataset), self._output_dim),
                    dtype=np.float32,
                    loader=loader_text,
                    obj=text_embeddings
                )

        return BaseOutput(embeddings=text_embeddings)


    @synchronize
    @torch.no_grad()
    def encode_query(self, loader_query:DataLoader, load_all_encode:bool=False):
        """
        Encode each query into a vector.

        Args:
            load_all_encode: bool, set to true to load the entire cache file

        Returns:
            BaseOutput:
                query_embeddings: array of [N, D]
        """
        query_embedding_path = os.path.join(self.query_dir, "query_embeddings.mmp")

        if load_all_encode:
            query_embeddings = np.memmap(
                query_embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_query.dataset), self._output_dim)

        elif self.config.load_encode or self.config.load_query_encode:
            query_embeddings = np.memmap(
                query_embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(loader_query.dataset), self._output_dim)[loader_query.sampler.start: loader_query.sampler.end]

        else:
            query_embeddings = np.zeros((len(loader_query.sampler), self._output_dim), dtype=np.float32)
            start_idx = end_idx = 0
            self.logger.info(f"encoding {self.config.dataset} {self.config.eval_set} query...")
            for i, x in enumerate(tqdm(loader_query, leave=False, ncols=100)):
                query_embedding = self.encode_query_step(x) # B, D

                end_idx += query_embedding.shape[0]
                query_embeddings[start_idx: end_idx] = query_embedding
                start_idx = end_idx
                if self.config.debug:
                    if i > 10:
                        break

            if self.config.save_encode:
                self.save_to_mmp(
                    path=query_embedding_path,
                    shape=(len(loader_query.dataset), self._output_dim),
                    dtype=np.float32,
                    loader=loader_query,
                    obj=query_embeddings
                )

        return BaseOutput(embeddings=query_embeddings)


    def faiss_index(self, loader_text:DataLoader):
        """
        Construct :class:`utils.index.FaissIndex`
        """
        text_embeddings = None
        if not self.config.load_index:
            text_embeddings = self.encode_text(loader_text).embeddings

        if self.config.index_type != "Flat" and not self.config.is_main_proc > 0:
            index = None
        else:
            if self.config.device != "cpu":
                # release temperary gpu cache so that faiss can use it
                torch.cuda.empty_cache()

            index = FaissIndex(
                index_type=self.config.index_type,
                d=self._output_dim,
                metric=self.config.dense_metric,
                start_text_idx=loader_text.sampler.start,
                device=self.config.device,
                save_dir=self.index_dir,
            )
            if self.config.load_index:
                index.load()

            index.fit(text_embeddings)

            if self.config.save_index:
                index.save()

        return BaseOutput(index=index)


    @synchronize
    def index(self, loaders):
        """
        Wrapper to construct a variety of faiss indexes. Subclass may override this function to create customized index.
        """
        return self.faiss_index(loaders["text"])


    @synchronize
    def retrieve(self, loaders:LOADERS) -> RETRIEVAL_MAPPING:
        """
        #. Retrieve by the index;

        #. Compute auxillary metrics if necessary;

        #. Post verify if necessary.

        Returns:
            retrieval result
        """
        loader_query = loaders["query"]

        index = self.index(loaders).index

        # place the encode_query outside of the if condition because there is a synchronize step inside encode_query function
        encode_output = self.encode_query(loader_query)
        query_embeddings = encode_output.embeddings

        if index is not None:
            t1 = time.time()
            self.logger.info("searching...")

            if "Flat" in index.name:
                verifier = self.init_verifier(loaders)
            else:
                # load all verifier for ANN indexes like IVFPQ, since it only stores at rank==0
                verifier = self.init_verifier(loaders, load_all_verifier=True)

            retrieval_result, posting_list_length = index.search(
                query_embeddings=query_embeddings,
                hits=self.config.hits,
                eval_posting_length=self.config.eval_posting_length and "IVF" in self.config.index_type,
                # the following config are index-specific, may be missing
                nprobe=self.config.get("nprobe"),
                efSearch=self.config.get("hnswef"),
                verifier=verifier
            )
            t2 = time.time()
            # manually delete the index
            del index

            if self.config.eval_posting_length and posting_list_length:
                # ANN index does not support parallel
                # if self.config.is_distributed:
                #     posting_list_length = np.asarray(self._gather_objects(posting_list_length)).sum()
                self.metrics["Posting_List_Length"] = int(np.round(posting_list_length))
                self.logger.info(f"Average Posting Length is {self.metrics['Posting_List_Length']}!")
        else:
            retrieval_result = defaultdict(list)

        return retrieval_result


    # TODO: review this function
    @torch.no_grad()
    def cluster(self, loaders:LOADERS):
        """Perform clusering over cached embeddings.
        """
        from utils.util import Cluster
        assert not self.config.is_distributed, "Clustering only available when not distributed!"
        assert "-" in self.config.cluster_type, "Use hyphen to separate cluster type and cluster metric"
        cluster_type, cluster_metric = self.config.cluster_type.split("-")

        if cluster_type == "flat":
            self.logger.info(f"{self.config.cluster_type} clustering text embeddings...")
            cluster_num = self.config.ncluster
            cluster_dir = os.path.join(self.config.cache_root, "cluster", self.name, self.config.cluster_type, str(cluster_num))
            os.makedirs(cluster_dir, exist_ok=True)

            loader_text = loaders["text"]
            encode_output = self.encode_text(loader_text)
            text_embeddings = encode_output.embeddings
            num_replicas = 50

            cluster = Cluster(device=self.config.device)

            assignments = cluster.kmeans(text_embeddings, cluster_num, num_replicas=num_replicas, metric=cluster_metric)
            centroids = cluster.get_centroids()

            np.save(os.path.join(cluster_dir, "centroids.npy"), centroids)

            assignments_mmp = np.memmap(
                os.path.join(cluster_dir, "assignments.mmp"),
                shape=(text_embeddings.shape[0], num_replicas),
                mode="w+",
                dtype=np.int32
            )
            assignments_mmp[:] = assignments

            # compute node number per cluster
            cluster_node_num = [0]*cluster_num
            for x in assignments[:, 0]:
                cluster_node_num[x] += 1
            cluster_node_num = np.asarray(cluster_node_num)
            self.logger.info(f"clustered {len(text_embeddings)} nodes into {len(centroids)} clusters, average cluster node number is {cluster_node_num.mean()}, max cluster node number is {cluster_node_num.max()}, min cluster node number is {cluster_node_num.min()}")

        elif cluster_type == "hier":
            loader_text = loaders["text"]
            encode_output = self.encode_text(loader_text)
            text_embeddings = encode_output.embeddings

            cluster_dir = os.path.join(self.config.cache_root, "cluster", self.name, self.config.cluster_type)
            os.makedirs(cluster_dir, exist_ok=True)
            cluster = Cluster(device=self.config.device)

            cluster_num = self.config.ncluster
            assignments = cluster.hierarchical_kmeans(text_embeddings, cluster_num, self.config.nleaf, metric=cluster_metric)
            # assignments = load_pickle("assignments.pkl")
            all_code_length = np.array([len(x) for x in assignments])
            self.logger.info(f"average code length is {all_code_length.mean()}, max code length is {all_code_length.max()}, min code length is {all_code_length.min()}")
            # save_pickle(assignments, "assignments.pkl")

            assignments_mmp = np.memmap(
                os.path.join(cluster_dir, "assignments.mmp"),
                shape=(text_embeddings.shape[0], all_code_length.max()),
                mode="w+",
                dtype=np.int32
            )
            assignments_mmp[:] = -1
            for i, x in enumerate(assignments):
                assignments_mmp[i, :len(x)] = x
            del assignments_mmp

        elif cluster_type == "ivf":
            self.logger.info(f"{self.config.cluster_type} clustering text embeddings...")
            cluster_num = self.config.ncluster
            cluster_dir = os.path.join(self.config.cache_root, "cluster", self.name, self.config.cluster_type, str(cluster_num))
            os.makedirs(cluster_dir, exist_ok=True)

            loader_text = loaders["text"]
            encode_output = self.encode_text(loader_text)
            text_embeddings = encode_output.embeddings
            num_replicas = 50

            ivf = faiss.index_factory(self._output_dim, f"IVF{cluster_num},Flat", faiss.METRIC_INNER_PRODUCT if cluster_metric == "ip" else faiss.METRIC_L2)
            if self.config.device != "cpu":
                ivf = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.config.device, ivf)

            ivf.train(text_embeddings)
            quantizer = faiss.downcast_index(ivf.quantizer)

            if self.config.device != "cpu":
                centroids = faiss.rev_swig_ptr(faiss.index_gpu_to_cpu(quantizer).get_xb(), quantizer.ntotal * quantizer.d).reshape(quantizer.ntotal, quantizer.d)
            else:
                centroids = faiss.rev_swig_ptr(quantizer.get_xb(), quantizer.ntotal * quantizer.d).reshape(quantizer.ntotal, quantizer.d)

            np.save(os.path.join(cluster_dir, "centroids.npy"), centroids)

            assignments = np.memmap(
                os.path.join(cluster_dir, "assignments.mmp"),
                shape=(text_embeddings.shape[0], num_replicas),
                mode="w+",
                dtype=np.int32
            )

            batch_size = 1000
            for i in range(0, text_embeddings.shape[0], batch_size):
                q = text_embeddings[i: i + batch_size]
                score, assignment = quantizer.search(q, num_replicas)
                assignments[i: i + batch_size] = assignment

            # compute node number per cluster
            cluster_node_num = [0]*cluster_num
            for x in assignments[:, 0]:
                cluster_node_num[x] += 1
            cluster_node_num = np.asarray(cluster_node_num)
            self.logger.info(f"clustered {len(text_embeddings)} nodes into {len(centroids)} clusters, average cluster node number is {cluster_node_num.mean()}, max cluster node number is {cluster_node_num.max()}, min cluster node number is {cluster_node_num.min()}")


    @torch.no_grad()
    def generate_code(self, loaders:LOADERS):
        """
        Generate codes from the cached clusering assignments.
        """
        if self.config.is_main_proc:
            # the code is bind to the code_tokenizer
            code_path = os.path.join(self.config.cache_root, "codes", self.config.code_type, self.config.code_tokenizer, str(self.config.code_length), "codes.mmp")
            # all codes are led by 0 and padded by -1
            self.logger.info(f"generating codes from {self.config.code_type} with code_length: {self.config.code_length}, saving at {code_path}...")

            loader_text = loaders["text"]
            text_num = len(loader_text.dataset)
            assignment_path = os.path.join(self.config.cache_root, "cluster", self.name, self.config.cluster_type, "assignments.mmp")

            # generate codes from pre-defined cluster assignments
            if os.path.exists(assignment_path):
                makedirs(code_path)
                assignments = np.memmap(
                    assignment_path,
                    mode="r+",
                    dtype=np.int32,
                ).reshape(text_num, -1)

                assert self.config.code_length >= assignments.shape[1] + 2, "The code_length must be greater than the assignment length by 2 because we have a leading 0 and an eos_token_id!"
                text_codes = np.memmap(
                    code_path,
                    # plus one because the code should be lead with the padding token id
                    shape=(text_num, self.config.code_length),
                    mode="w+",
                    dtype=np.int32
                )
                tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.config.plm_root, self.config.code_tokenizer))
                model = AutoModel.from_pretrained(os.path.join(self.config.plm_root, self.config.code_tokenizer))
                try:
                    start_token_id = model._get_decoder_start_token_id()
                except ValueError:
                    start_token_id = model.config.pad_token_id
                    self.logger.warning(f"Decoder start token id not found, use pad token id ({start_token_id}) instead!")

                # the codes are always led by start_token_id and padded by -1
                text_codes[:, 0] = start_token_id
                text_codes[:, 1:] = -1

                bias = tokenizer.vocab_size
                # another bias to distinguish the same cluster id in different layer
                if self.config.code_type.split("-")[-1] == "bias":
                    bias += np.arange(text_codes.shape[1]) * (assignments.max() + 1)

                for i, x in enumerate(assignments):
                    length = (x != -1).sum()
                    text_codes[i, 1: length + 1] = x[:length]
                    if isinstance(bias, np.ndarray):
                        text_codes[i, 1: length + 1] += bias[:length]
                    else:
                        text_codes[i, 1: length + 1] += bias

                    # assign eos_token_id
                    text_codes[i, length + 1] = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id

            else:
                raise FileNotFoundError(f"{assignment_path} not found!")



class BaseGenerativeModel(BaseModel):
    """
    Base class for generative models e.g. DSI, WebUltron.
    """
    def __init__(self, config:Config):
        super().__init__(config)
        #: str: we separate the saving folder of generative model
        self.code_dir = os.path.join(self.config.cache_root, "codes", self.name if self.config.code_type == "self" else self.config.code_type, self.config.code_tokenizer, str(self.config.code_length))


    @synchronize
    @torch.no_grad()
    def encode_text(self, loader_text:DataLoader):
        """
        Encode each text into its code.
        """
        text_codes = loader_text.dataset.text_codes[loader_text.sampler.start: loader_text.sampler.end]
        return BaseOutput(codes=text_codes)


    def generative_index(self, loader_text:DataLoader):
        """
        Construct :class:`utils.index.TrieIndex`.
        """
        encode_output = self.encode_text(loader_text)    # N, L
        text_codes = encode_output.codes

        tokenizer = AutoTokenizer.from_pretrained(self.config.plm_dir)

        index = GENERATIVE_INDEX_MAP[self.config.index_type](
            rank=self.config.rank,
            save_dir=self.code_dir,
            pad_token_id=self.config.special_token_ids["pad"][1],
            eos_token_id=self.config.special_token_ids["eos"][1],
            sep_token_id=tokenizer.convert_tokens_to_ids(self.config.code_sep) if self.config.get("code_sep") is not None else None,
        )

        index.fit(
            text_codes=text_codes,
            tokenizer=tokenizer,
            load_index=self.config.load_index,
            # only save at rank==0 because tries across processes are always the same
            save_index=self.config.save_index,
            threads=self.config.get("index_thread"),
            shards=self.config.get("index_shard"),
            separator=self.config.get("code_sep")
        )

        return BaseOutput(index=index)


    @synchronize
    def index(self, loaders:LOADERS):
        """
        Wrapper to construct a variety of trie indexes. Subclass may override this function to create customized index.
        """
        if self.config.index_type in GENERATIVE_INDEX_MAP:
            return self.generative_index(loaders["text"])


    @synchronize
    @torch.no_grad()
    def retrieve(self, loaders:LOADERS) -> RETRIEVAL_MAPPING:
        """
        #. Retrieve by the index;

        #. Save the generated query codes if necessary;

        #. Compute auxillary metrics if necessary;

        #. Post verify if necessary.

        Returns:
            retrieval result
        """
        index = self.index(loaders).index
        loader_query = loaders["query"]

        retrieval_result = {}

        self.logger.info("searching...")
        start_idx = 0
        # in case the query is parallel
        query_start_idx = loader_query.sampler.start

        beam_decoder = BeamDecoder()

        tokenizer = AutoTokenizer.from_pretrained(self.config.plm_dir)
        # new_query_file = open(f"queries.autotsg.tsv", "w")

        for i, x in enumerate(tqdm(loader_query, leave=False, ncols=100)):
            # if not (x["query_idx"].unsqueeze(-1) == torch.tensor([1466])).any():
            #     continue

            query = self._move_to_device(x["query"])
            encoder_outputs = self.plm.encoder(**query)
            B = query["input_ids"].shape[0]
            end_idx = start_idx + B

            beam_decoder.search(
                model=self.plm, 
                query={**query, "encoder_outputs": encoder_outputs},
                nbeam=self.config.nbeam, 
                threshold=self.config.beam_trsd, 
                trsd_start_len=self.config.trsd_start_len, 
                max_new_tokens=self.config.code_length - 1, 
                constrain_index=index,
                rank_type=self.config.rank_type,
                tokenizer=tokenizer,
                do_sample=self.config.decode_do_sample,
                do_greedy=self.config.decode_do_greedy,
                topk=self.config.sample_topk,
                topp=float(self.config.sample_topp) if self.config.sample_topp is not None else None,
                typical_p=float(self.config.sample_typicalp) if self.config.sample_typicalp is not None else None,
                temperature=float(self.config.sample_tau) if self.config.sample_tau is not None else None,
                renormalize_logits=self.config.decode_renorm_logit,
                do_early_stop=self.config.get("wordset_early_stop"),
                early_stop_start_len=self.config.get("early_stop_start_len"),
            )
            beams = beam_decoder.beams
            eos_hidden_states = beam_decoder.eos_hidden_states

            # # write to query file
            # first_beam = [beam[0] for beam in beams]
            # query_token_id = x["query"]["input_ids"]
            # query_idx = x["query_idx"].tolist()
            # for q, b, idx in zip(query_token_id, first_beam, query_idx):
            #     b = [c for c in b if c != index.sep_token_id]
            #     q = tokenizer.decode(q, skip_special_tokens=True)
            #     b = tokenizer.decode(b, skip_special_tokens=True)
            #     line = str(idx) + "\t" + q + " " + b + "\n"
            #     new_query_file.write(line)

            # ranking by score
            if self.config.rank_type == "eos":
                eos_hidden_states = torch.stack(sum(eos_hidden_states, []), dim=0)
                scores = self.scorer(eos_hidden_states).squeeze(-1).tolist()
            elif self.config.rank_type == "prob":            
                # ranking by generation prob
                scores = sum(beam_decoder.seq_scores, [])
            else:
                raise NotImplementedError(f"Ranking type {self.config.ranking_type} is not implemented yet!")

            offset = 0
            for j, batch in enumerate(beams):
                res = defaultdict(list)
                for k, c in enumerate(batch):
                    # need to provide prev_text_indices
                    ids = beam_decoder.prev_text_indices[j][k]
                    for id in ids:
                        res[id].append(scores[offset + k])
                offset += len(batch)
                retrieval_result[j + start_idx + query_start_idx] = [(k, max(v)) for k, v in res.items()]

            start_idx = end_idx

            if self.config.get("debug") and i > 1:
                break

        # new_query_file.close()
        return retrieval_result


# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from fvcore.common.timer import Timer
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
import cv2
from tqdm import tqdm
"""
This file contains functions to parse human coherence demo annotations into dicts in the
"Detectron2 format".
"""

LOCAL_HUMAN_ARTIFACT_CATEGORIES = [
    {'id': 1, 'class_name': 'face', 'synonyms': []},
    {'id': 2, 'class_name': 'torso', 'synonyms': []},
    {'id': 3, 'class_name': 'arm', 'synonyms': []},
    {'id': 4, 'class_name': 'leg', 'synonyms': []},
    {'id': 5, 'class_name': 'hand', 'synonyms': []},
    {'id': 6, 'class_name': 'feet', 'synonyms': []},
]

GLOBAL_HUMAN_ARTIFACT_CATEGORIES = [
    {'id': 1, 'class_name': 'human missing arm', 'synonyms': []},
    {'id': 2, 'class_name': 'human missing face', 'synonyms': []},
    {'id': 3, 'class_name': 'human missing feet', 'synonyms': []},
    {'id': 4, 'class_name': 'human missing hand', 'synonyms': []},
    {'id': 5, 'class_name': 'human missing leg', 'synonyms': []},
    {'id': 6, 'class_name': 'human missing torso', 'synonyms': []},
    {'id': 7, 'class_name': 'human with extra arm', 'synonyms': []},
    {'id': 8, 'class_name': 'human with extra face', 'synonyms': []},
    {'id': 9, 'class_name': 'human with extra feet', 'synonyms': []},
    {'id': 10, 'class_name': 'human with extra hand', 'synonyms': []},
    {'id': 11, 'class_name': 'human with extra leg', 'synonyms': []},
    {'id': 12, 'class_name': 'human with extra torso', 'synonyms': []},
]

logger = logging.getLogger(__name__)

__all__ = ["load_human_artifact_json", "register_local_human_artifact_instances", "get_local_human_artifact_instances_meta", "register_global_human_artifact_instances", "get_global_human_artifact_instances_meta"]


def register_local_human_artifact_instances(name, metadata, json_root, image_root):
    """
    Register a dataset in human coherence demo's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "human_artifact_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_root (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_human_artifact_json(json_root, image_root, name))
    MetadataCatalog.get(name).set(
        json_root=json_root, image_root=image_root, evaluator_type="local_human_artifact", **metadata
    )

def register_global_human_artifact_instances(name, metadata, json_root, image_root):
    """
    Register a dataset in human coherence demo's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "human_artifact_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_root (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_human_artifact_json(json_root, image_root, name, global_artifact=True))
    MetadataCatalog.get(name).set(
        json_root=json_root, image_root=image_root, evaluator_type="global_human_artifact", **metadata
    )

def load_human_artifact_json(json_root, image_root, dataset_name=None, extra_annotation_keys=None, global_artifact=False):
    """
    Load a json file in human coherence demo's annotation format.

    Args:
        json_root (str): full path to the human coherence demo json annotation file.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "human_artifact_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "bbox", "bbox_mode", "category_id",
            "segmentation"). The values for these keys will be returned as-is.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    timer = Timer()
    # lvis_api = LVIS(json_file)
    image_file = sorted(os.listdir(image_root))
    if timer.seconds() > 1:
        logger.info("Loading {} files takes {:.2f} seconds.".format(len(image_file), timer.seconds()))

    if dataset_name is not None:
        meta = get_local_human_artifact_instances_meta(dataset_name) if not global_artifact else get_global_human_artifact_instances_meta(dataset_name)
        MetadataCatalog.get(dataset_name).set(**meta)

    dataset_dicts = []
    for idx, v in enumerate(image_file):
        record = {}
        filename = os.path.join(image_root, v)
        ext = os.path.splitext(filename)[1]
        height, width = cv2.imread(filename).shape[:2]
        anno_file = os.path.join(json_root, v.replace(ext, ".json"))
        anno = json.load(open(anno_file))
        record["file_name"] = filename
        record["anno_name"] = anno_file
        record["image_id"] = idx
        record["height"] = anno['image']['height']
        record["width"] = anno['image']['width']

        objs = []
        if not global_artifact:
            annos = anno["annotation"]
            for anno in annos:
                lbl = anno["body_parts"]
                if anno["level"] == "mild" and "only_severe" in dataset_name.lower():
                    continue
                if lbl not in meta["look_up"]:
                    continue
                bbox = anno["bbox"]
                if bbox[0] > bbox[2]:
                    bbox[0], bbox[2] = bbox[2], bbox[0]
                if bbox[1] > bbox[3]:
                    bbox[1], bbox[3] = bbox[3], bbox[1]
                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    # "segmentation": [poly],
                    "category_id": meta["look_up"][lbl]-1,
                }
                objs.append(obj)
        else:
            annos = anno["human"]
            for anno in annos:
                tags = anno["tag"]
                for t in tags:
                    if t not in meta["look_up"]:
                        continue
                    lbl = meta["look_up"][t]-1
                    bbox = anno["bbox"]
                    if bbox[0] > bbox[2]:
                        bbox[0], bbox[2] = bbox[2], bbox[0]
                    if bbox[1] > bbox[3]:
                        bbox[1], bbox[3] = bbox[3], bbox[1]

                    obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": lbl,
                    }
                    objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def get_local_human_artifact_instances_meta(dataset_name):
    """
    Load human coherence demo metadata.

    Args:
        dataset_name (str): human coherence demo dataset name without the split name (e.g., "human_artifact_unary").

    Returns:
        dict: human coherence demo metadata with keys: thing_classes
    """
    return _get_local_human_artifact_meta()

def get_global_human_artifact_instances_meta(dataset_name):
    """
    Load human coherence demo metadata.

    Args:
        dataset_name (str): human coherence demo dataset name without the split name (e.g., "human_artifact_unary").

    Returns:
        dict: human coherence demo metadata with keys: thing_classes
    """
    return _get_global_human_artifact_meta()
    
def _get_local_human_artifact_meta():
    assert len(LOCAL_HUMAN_ARTIFACT_CATEGORIES) == 6
    cat_ids = [k["id"] for k in LOCAL_HUMAN_ARTIFACT_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    human_artifact_categories = sorted(LOCAL_HUMAN_ARTIFACT_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["class_name"] for k in human_artifact_categories]
    look_up = {k["class_name"]: k["id"] for k in human_artifact_categories}
    for k in human_artifact_categories:
        if "synonyms" in k:
            for syn in k["synonyms"]:
                look_up[syn] = k["id"]
    meta = {"thing_classes": thing_classes, "look_up": look_up}
    return meta

def _get_global_human_artifact_meta():
    assert len(GLOBAL_HUMAN_ARTIFACT_CATEGORIES) == 12
    cat_ids = [k["id"] for k in GLOBAL_HUMAN_ARTIFACT_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    human_artifact_categories = sorted(GLOBAL_HUMAN_ARTIFACT_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["class_name"] for k in human_artifact_categories]
    look_up = {k["class_name"]: k["id"] for k in human_artifact_categories}
    for k in human_artifact_categories:
        if "synonyms" in k:
            for syn in k["synonyms"]:
                look_up[syn] = k["id"]
    meta = {"thing_classes": thing_classes, "look_up": look_up}
    return meta

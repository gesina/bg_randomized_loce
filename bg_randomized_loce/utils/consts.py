"""Helpful constants for accessing results dataframes."""
import os
from typing import Literal, Union, Callable, get_args, Optional

import torch
from transformers import BaseImageProcessor

from ..data_structures.datasets import SegmentationDataset
from ..data_structures.folder_classification_datasets import PlacesDataset, SyntheticBackgroundsDataset
from ..data_structures.imagenet import ImageNetS50SegmentationDataset
from ..data_structures.imagenet_labels import IMAGENET_CLASS_ID_BY_NAME, IMAGENET_CLASS_NAME_BY_WORDNET_ID
from ..data_structures.mscoco import COCO_TO_WORDNET_IDS, VOC_TO_WORDNET_IDS
from ..data_structures.mscoco import MSCOCOSegmentationDataset
from ..hooks import Propagator
from ..loce.loce_utils import standardize_propagator, efficientnet_propagator_builder, vit_propagator_builder, \
    detr_propagator_builder, yolo5_propagator_builder, swin_propagator_builder, mobilenet_propagator_builder, \
    vgg_propagator_builder

# some key shortcuts to make it more convenient to type
CE_METHOD = 'ce_method'
BG = 'bg_randomizer_key'
TEST_BG = 'test_bg_key'
NUM_BG = 'num_bgs_per_ce'
DATA = 'dataset_key'
MODEL = 'model_key'
CAT = 'category_id'
LAYER = 'layer'
CE = 'ce'
RUN = 'run'
IMG_ID = 'img_id'
TEST_IMG_ID = 'test_img_id'
TEST_IMG_SUBID = 'test_img_id'
IOU = 'iou'


# some derived settings
DEPTH = 'depth'
GLO_OR_LOC = 'global_or_local'

# some value shortcuts
NET2VEC = 'net2vec_proper_bce'
NET2VEC_OLD = 'net2vec'
LOCE = 'loce_proper_bce'
LOCE_OLD = 'loce'
GLOCE = 'globalized_loce_proper_bce'
GLOCE_OLD = 'globalized_loce'
GLOBAL = 'global'
LOCAL = 'local'
LOC_TO_GLOB = 'local_to_global'
VANILLA = 'vanilla'

PRETTY_NAMES = {
    CE_METHOD: 'CE method',
    BG: 'train bg',
    TEST_BG: 'test bg',
    NUM_BG: '#bg/image',
    DATA: 'train data',
    MODEL: 'model',
    CAT: 'concept',
    LAYER: 'layer',
    CE: 'CE',
    IOU: 'IoU',
    GLO_OR_LOC: 'global/local',
    NET2VEC: 'Net2Vec', 
    LOCE: 'LoCE',
    GLOCE: 'GloCE',
    LOC_TO_GLOB: 'loc. to glob.',
    "places": "Places205",
    "places_voronoi": "Voronoi",
    "efficientnet": "EffNet",
    "vit": "ViT",
    "detr": "DETR",
    "swin": "SWIN",
    "mobilenet": "MobileNet",
    "vgg": "VGG16",
    "pascal_voc": "VOC",
    "imagenets50": "ImageNetS50",
    }

# The columns we are interested in (apart from the vector, which is 'ce')
GLOBAL_CE_METHODS = (NET2VEC_OLD, NET2VEC, GLOCE_OLD, GLOCE)
EXPERIMENT_SETTING_COLS = (RUN, CE_METHOD, BG, NUM_BG, DATA, MODEL, CAT, LAYER)
ADDITIONAL_COLS = ('loss', 'rel_path', 'abs_path')
DERIVED_SETTINGS = (DEPTH, GLO_OR_LOC)



_BG_DATASET_KEY = Literal["vanilla", "places", "places_voronoi", "synthetic",]
_MODEL_KEY = Literal['efficientnet', 'vit', 'detr', 'yolo', 'swin', 'mobilenet', 'vgg']
_CE_METHOD_KEY = Literal["loce_proper_bce", "net2vec_proper_bce", "net2vec", "loce",]
_DATASET_KEY = Literal["pascal_voc", "imagenets50"]

STORAGE_TEMPLATE: str = os.path.join("{run}",
                                     "{ce_method}",
                                     "{bg_randomizer_key}",
                                     "{num_bgs_per_ce}_bgs_per_ce",
                                     "{dataset_key}",
                                     "loce_*_{model_key}",
                                     "{category_id}_{img_id}.pkl")



DATA_SPLITS: dict[_DATASET_KEY, dict[Union[str, int], dict[Literal['train', 'test'], list[Union[str, int]]]]] = {
    "pascal_voc": {
        1: {"train": [1947, 1140, 896, 1992, 1142, 1155, 1707, 2538, 48, 2478, 127, 1244, 1429, 886, 1392, 1784, 1625,
                      1976, 465, 1609, 27, 1612, 2632, 2008, 259, 1648, 2182, 1484, 2191, 1950, 40, 2694, 2724, 2232,
                      1929, 129, 2785, 2152, 2639, 1722, 1538, 2515, 2337, 134, 1935, 2299, 2263, 2, 2504, 2129],
            "test": [2118, 1812, 1809, 871, 1081, 2207, 725, 1843, 1366, 2640, 2759, 1720, 2505, 2630, 1279, 806, 281,
                     1043,
                     2835, 1205]},
        2: {"train": [1651, 1146, 1590, 1666, 1109, 10, 1113, 343, 1023, 1871, 403, 2779, 1801, 314, 1637,
                      2085, 2784, 1294, 332, 1460, 2283, 2013, 1829, 731, 618, 1730, 789, 1832, 2168, 788,
                      2043, 355, 1639, 95, 36, 1418, 2015, 1444, 2411, 2099, 2174, 774, 572, 2300, 1535, 464,
                      137, 1687, 1292, 1266],
            "test": [805, 1895, 784, 1614, 35, 2201, 1860, 2471, 2845, 1112, 2274, 1326, 867, 341, 1559, 931,
                     2205, 2031, 945, 205]},
        3: {"train": [131, 2111, 2805, 927, 2698, 743, 1837, 1340, 2678, 215, 2645, 1440, 2801, 1408, 198,
                      2768, 2355, 154, 1269, 1382, 230, 315, 2094, 218, 2852, 540, 505, 1925, 1386, 1347,
                      1683, 2329, 1802, 2900, 320, 2685, 803, 1401, 1192, 1867, 2586, 2401, 1137, 442, 649,
                      1473, 2425, 2354, 1372, 1997],
            "test": [329, 1916, 239, 1428, 2744, 2550, 2231, 1529, 841, 1506, 1027, 1330, 1814, 2541, 2270,
                     853, 135, 2546, 644, 615]},
        4: {"train": [417, 1839, 1632, 809, 1945, 90, 1494, 2556, 2719, 1148, 2863, 986, 958, 1688, 1977,
                      2215, 1145, 665, 1710, 2882, 647, 1132, 1378, 2376, 419, 250, 1391, 1863, 921, 631,
                      1539, 531, 2603, 2252, 1086, 1667, 1387, 1562, 516, 2511, 2773, 1772, 1162, 948, 2438,
                      461, 2888, 2564, 1565, 271],
            "test": [153, 699, 883, 1309, 2749, 44, 590, 2475, 751, 1899, 1084, 1350, 1323, 466, 2375, 1080,
                     754, 778, 999, 378]},
        5: {"train": [297, 439, 1521, 2201, 2582, 93, 2411, 2623, 2305, 2253, 2242, 2531, 1889, 333, 11,
                      2893, 1457, 1679, 440, 2145, 2489, 487, 2890, 2031, 634, 343, 98, 2781, 2649, 70, 1629,
                      2638, 2714, 162, 1236, 1949, 1139, 1624, 2913, 1082, 2080, 1658, 2624, 1766, 405, 16,
                      400, 2439, 1560, 21],
            "test": [1289, 275, 541, 2297, 381, 1431, 1337, 282, 1338, 736, 2384, 2574, 925, 2197, 2146, 455,
                     2584, 469, 34, 1646]},
        6: {"train": [438, 224, 2741, 2559, 1906, 395, 1325, 1881, 1243, 2602, 2361, 2118, 580, 950, 2557,
                      2635, 556, 1168, 1715, 1320, 201, 97, 2060, 2226, 2431, 2881, 2682, 1479, 1333, 1395,
                      499, 1480, 539, 467, 1267, 1805, 1694, 2067, 242, 1958, 476, 122, 1102, 262, 46, 117,
                      1828, 1247, 459, 2706],
            "test": [295, 491, 52, 1974, 2359, 328, 2180, 2417, 2663, 40, 2526, 195, 2544, 979, 210, 161,
                     372, 1173, 1661, 2658]},
        7: {"train": [295, 2275, 2341, 1622, 2240, 1893, 233, 1418, 1545, 539, 1415, 1332, 1326, 945, 2513,
                      2774, 1658, 814, 2567, 301, 1995, 1673, 2047, 2667, 1313, 1002, 1356, 385, 2184, 2519,
                      772, 1063, 2328, 2434, 967, 2108, 780, 2881, 2897, 1790, 115, 2134, 2155, 1289, 224,
                      2036, 302, 556, 558, 2898],
            "test": [1525, 1165, 1240, 2732, 574, 1499, 435, 2725, 2241, 2226, 662, 542, 255, 698, 1228,
                     2730, 2470, 1005, 2560, 1013]},
        8: {"train": [2024, 1616, 1617, 1200, 1149, 2628, 2427, 33, 2172, 1178, 2651, 2654, 2456, 1336, 196,
                      1942, 2634, 226, 1421, 1220, 960, 612, 2004, 1717, 2287, 741, 462, 831, 415, 203, 305,
                      2098, 2042, 31, 2142, 2324, 304, 2717, 2435, 2259, 1284, 2029, 110, 2390, 2629, 965,
                      588, 1902, 2718, 2127],
            "test": [1413, 1490, 1613, 112, 1196, 2224, 1944, 2625, 2686, 2350, 176, 159, 2268, 1954, 2212,
                     2237, 2495, 2272, 2578, 2279]},
        9: {"train": [2878, 2788, 413, 1048, 381, 1311, 54, 2769, 729, 2671, 2536, 546, 884, 440, 2227, 2729,
                      1209, 2141, 2851, 793, 795, 2634, 1967, 2176, 570, 2843, 2487, 1579, 229, 2105, 2813,
                      2318, 2331, 1487, 1540, 1764, 545, 2896, 2713, 351, 106, 720, 1766, 1841, 1792, 1891,
                      2795, 2605, 1492, 589],
            "test": [2806, 1681, 2357, 1830, 2248, 2857, 118, 744, 2324, 1127, 2280, 1836, 2078, 2228, 463,
                     1750, 2351, 1727, 1271, 1003]},
        16: {"train": [2816, 1396, 818, 835, 2647, 1174, 172, 1724, 150, 1541, 2890, 872, 225, 2498, 1946,
                       1761, 1853, 602, 2135, 1695, 2383, 1463, 478, 1838, 2015, 2366, 613, 297, 2540, 2823,
                       413, 759, 2503, 545, 2371, 1766, 144, 1787, 509, 1846, 1953, 868, 2302, 1212, 2512,
                       2738, 1708, 445, 82, 278],
             "test": [1025, 1317, 2754, 1489, 1256, 970, 1582, 1458, 191, 1924, 1343, 2675, 1840, 2896, 188,
                      1475, 2256, 1083, 2588, 1920]},
        17: {"train": [1934, 1477, 299, 1036, 453, 811, 2367, 2491, 1973, 2786, 573, 1144, 428, 171, 1446,
                       2659, 1348, 1215, 2107, 2854, 2356, 2053, 1990, 116, 2170, 207, 180, 108, 1052, 1412,
                       1124, 1903, 436, 1474, 1999, 434, 345, 2581, 63, 420, 2751, 1621, 241, 2457, 2122,
                       1024, 1281, 309, 2014, 2101],
             "test": [1410, 1379, 2840, 1922, 1202, 1527, 474, 83, 2103, 2319, 2636, 166, 2208, 1498, 1097,
                      2258, 1861, 1438, 2826, 2251]},
        18: {"train": [110, 222, 2660, 1420, 386, 2092, 93, 825, 2380, 965, 1540, 2150, 2071, 2253, 2867,
                       1469, 36, 325, 2601, 267, 2587, 2804, 2677, 670, 2109, 2545, 41, 2004, 2675, 2896,
                       2264, 593, 2026, 710, 2890, 2539, 1617, 1759, 1285, 2464, 1239, 1265, 2543, 379, 607,
                       2498, 2912, 2043, 2176, 1458],
             "test": [2760, 1711, 2371, 1012, 351, 2212, 529, 2880, 2162, 1815, 2750, 330, 1407, 1176, 1570,
                      2872, 398, 545, 1791, 2823]},
        19: {"train": [353, 1170, 1965, 1819, 448, 2648, 2044, 1342, 1040, 350, 1825, 2365, 1306, 2483, 1034,
                       1009, 38, 1186, 1264, 716, 1534, 1226, 1608, 1888, 1520, 204, 2563, 598, 9, 294, 285,
                       2566, 2000, 1164, 2596, 2802, 2762, 2684, 2223, 1662, 1770, 1198, 2901, 1018, 1644,
                       2178, 249, 2163, 340, 1276],
             "test": [163, 1763, 2838, 2289, 2090, 2653, 273, 1486, 1015, 768, 926, 1530, 2068, 2236, 175,
                      2309, 1865, 390, 1094, 1880]},
        20: {"train": [1606, 1089, 551, 269, 514, 445, 2872, 235, 1329, 1571, 1138, 1547, 1600, 1231, 106,
                       589, 2576, 2305, 2739, 379, 503, 130, 274, 818, 919, 597, 1495, 368, 2495, 1311, 104,
                       838, 677, 2432, 222, 2763, 2386, 2473, 2031, 939, 603, 2611, 1003, 572, 968, 1672,
                       877, 455, 2910, 13],
             "test": [86, 2837, 1567, 546, 1351, 681, 1603, 1011, 961, 593, 451, 1072, 3, 1061, 1433, 1620,
                      2106, 833, 8, 270]},
        21: {"train": [], "test": []},
        22: {"train": [], "test": []},
        23: {"train": [], "test": []},
        24: {"train": [], "test": []},
        25: {"train": [], "test": []}, }
}

PROJECT_DIR = ''

DATA_BUILDERS: dict[_DATASET_KEY, tuple[type(SegmentationDataset), Callable[[...], SegmentationDataset]]] = {
    "pascal_voc": (MSCOCOSegmentationDataset.ALL_CAT_NAMES_BY_ID.items(),
                   lambda category_ids, **kwargs: MSCOCOSegmentationDataset(
                       **{**dict(
                           imgs_path=f"{PROJECT_DIR}/data/voc2012/VOCdevkit/VOC2012/JPEGImages/",
                           all_annots_path=f"{PROJECT_DIR}/data/voc2012/VOCdevkit/VOC2012/voc2012trainval_annotations.json",
                           category_ids=category_ids, ),
                          **kwargs}
                   ).subselect([i for c_id in category_ids for i in DATA_SPLITS["pascal_voc"][c_id]["train"]])
                   ),
    "imagenets50": (ImageNetS50SegmentationDataset.ALL_CAT_NAMES_BY_ID.items(),
                    lambda category_ids, **kwargs: ImageNetS50SegmentationDataset(
                        **{**dict(
                            imgs_path=f'{PROJECT_DIR}/data/ImageNetS/ImageNetS50/train',
                            masks_path=f'{PROJECT_DIR}/data/ImageNetS/ImageNetS50/train-semi-segmentation',
                            category_ids=category_ids),
                           **kwargs}),
                    ),
}
assert set(DATA_BUILDERS.keys()) == set(get_args(_DATASET_KEY))


TEST_DATA_BUILDERS: dict[_DATASET_KEY, Callable[[...], SegmentationDataset]] = {
    "pascal_voc": (lambda category_ids, **kwargs: MSCOCOSegmentationDataset(
        imgs_path=f"{PROJECT_DIR}/data/voc2012/VOCdevkit/VOC2012/JPEGImages/",
        all_annots_path=f"{PROJECT_DIR}/data/voc2012/VOCdevkit/VOC2012/voc2012trainval_annotations.json",
        category_ids=category_ids,
        **kwargs).subselect([i for c_id in category_ids for i in DATA_SPLITS["pascal_voc"][c_id]["test"]])
                   ),
    "imagenets50": (lambda category_ids, **kwargs: ImageNetS50SegmentationDataset(
        imgs_path=f'{PROJECT_DIR}/data/ImageNetS/ImageNetS50/validation',
        masks_path=f'{PROJECT_DIR}/data/ImageNetS/ImageNetS50/validation-segmentation',
        category_ids=category_ids,
        **kwargs)
                    ),
}

BG_DATA_BUILDERS: dict[_BG_DATASET_KEY, Callable[[...], Optional[SegmentationDataset]]] = {
    "vanilla": lambda **kwargs: None,
    "places": lambda **kwargs: PlacesDataset(imgs_path="./data/places_205_kaggle", **kwargs),
    "places_voronoi": lambda **kwargs: PlacesDataset(imgs_path="./data/places_205_kaggle", **kwargs),
    "synthetic": lambda **kwargs: SyntheticBackgroundsDataset(imgs_path="./data/synthetic_backgrounds", **kwargs),
}
assert set(BG_DATA_BUILDERS.keys()) == set(get_args(_BG_DATASET_KEY))


PLACES_SUBSETS = {
    "architecture": ["abbey", "aqueduct", "arch", "attic", "balcony_exterior", "basilica", "building_facade", "cathedral_outdoor", "embassy", "mosque_outdoor", "office_building", "synagogue_outdoor"],
    "indoors": ["alcove", "amusement_arcade", "bathroom", "bedchamber", "bedroom", "bow_window_indoor", "childs_room", "computer_room", "dining_room", "hotel_room", "kitchen", "kitchenette", "living_room"],
    #"pattern": ["anechoic_chamber", "ball_pit"],
    #"underwater": ["aquarium", "u_underwater_ocean_deep", "u_underwater_coral_reef"],
    "at_water": ["bayou", "beach", "canal_natural", "canyon", "coast", "creek", "dock", "islet", "lagoon", "lake_natural", "marsh", "ocean", "pond", "waterfall_plunge", "wave"],
    "machinery": ["auto_factory", "electrical_substation", "engine_room"],
    "open_lands": ["badlands", "butte", "desert_vegetation", "tundra"],
    #"desert_like": ["desert_sand", ],#"desert_road"],
    "forest": ["bamboo_forest", "forest_broadleaf", "forest_needleleaf", "forest_path", "rainforest", "woodland"],
    "botanical": [ "botanical_garden", "cottage_garden", "formal_garden", "japanese_garden", "orchard", "topiary_garden"],
    "field": ["field_cultivated", "golf_course", "hayfield", "wheat_field", "fairway", ],
    #"cabin": ["car_interior_backseat", "car_interior_frontseat", "airplane_cabin", "subway_interior", "train_interior"],
    "snow": [ "crevasse", "glacier", "ice_shelf", "iceberg", "mountain_snowy", "ski_slope", "snowfield"],
    "road": ["crosswalk", "freeway", "highway", "raceway", "street", "toll_plaza"],
    # LEFT OUT: sports (due to too many humans)
}

AS_IMAGENET_IDS_OR_NAMES = \
    IMAGENET_CLASS_ID_BY_NAME | \
    {c: [IMAGENET_CLASS_NAME_BY_WORDNET_ID[wnid] for wnid in wnids] for c, wnids in COCO_TO_WORDNET_IDS.items()} | \
    {c: [IMAGENET_CLASS_NAME_BY_WORDNET_ID[wnid] for wnid in wnids] for c, wnids in VOC_TO_WORDNET_IDS.items()}


WRAPPED_MODELS: dict[
    _MODEL_KEY, Callable[[list[str], Union[str, torch.device]], tuple[Propagator, 'BaseImageProcessor']]] = {
    "efficientnet": standardize_propagator(efficientnet_propagator_builder),
    "vit": standardize_propagator(vit_propagator_builder),
    "detr": standardize_propagator(detr_propagator_builder),
    "yolo": standardize_propagator(yolo5_propagator_builder),
    "swin": standardize_propagator(swin_propagator_builder),
    "mobilenet": standardize_propagator(mobilenet_propagator_builder),
    "vgg": standardize_propagator(vgg_propagator_builder)
}
assert set(WRAPPED_MODELS.keys()) == set(get_args(_MODEL_KEY))

LAYERS_BY_MODEL: dict[_MODEL_KEY, list[str]] = {
    "detr": ["model.backbone.conv_encoder.model.layer4", "model.input_projection", "model.encoder.layers.5"],
    "vit": ["conv_proj", "encoder.layers.encoder_layer_6", "encoder.layers.encoder_layer_11"],
    "swin": [
        "features.0",  # conv patches
        "features.3",  # transformer blocks (odd) with PatchMerging (even) inbetween
        "features.7"
    ],
    "efficientnet": ['features.4.2', 'features.6.0', 'features.7.0'],
    "mobilenet": [
        'features.7.block.2.0',
        'features.12.block.3.0',
        'features.16.0',  # last conv layer
    ],
    "yolo": ['4.cv3.conv', '14.conv', '23.cv3.conv'],
    "vgg": [
        'features.7',  # before the second MaxPool
        'features.21',  # before the 4th MaxPool
        'features.28',  # last conv layer (and before last MaxPool)
    ]
}
assert set(LAYERS_BY_MODEL.keys()) == set(get_args(_MODEL_KEY))

LAYERS_BY_DEPTH = {
    'early': [layers[0] for layers in LAYERS_BY_MODEL.values()],
    'middle': [layers[1] for layers in LAYERS_BY_MODEL.values()],
    'late': [layers[2] for layers in LAYERS_BY_MODEL.values()],
}

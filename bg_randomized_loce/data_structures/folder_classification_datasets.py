import os
from collections import defaultdict
from typing import Optional, TYPE_CHECKING, Any, Union

from .datasets import SegmentationDataset, ImageLoader

if TYPE_CHECKING:
    from .datasets import _TransformType
    import torch

if TYPE_CHECKING:
    _CatID = str
    _ImgID = str


class FolderClassificationDataset(SegmentationDataset):
    ALL_CAT_NAMES_BY_ID = {}
    DEFAULT_IMAGE_LOADER = ImageLoader(img_shape=None)
    DEFAULT_IMGS_PATH = None

    def __init__(self,
                 imgs_path: str = DEFAULT_IMGS_PATH,
                 *,
                 transform: Optional['_TransformType'] = None,
                 image_loader: ImageLoader = None,
                 device: Union['torch.device', str] = None,
                 category_ids: list[str] = None,
                 ):
        """
        Custom class to load a random image from a specific class in the Places Dataset.

        Args:
            imgs_path (str): Path to the root directory containing class folders.
                            The structure should be root/class_name/image.jpg
        """
        # Create a mapping from class names to image file paths
        self._img_infos_by_cat_id: dict['_CatID', list[dict[str, Any]]] = \
            self._build_class_to_images(imgs_path, allowed_categories=category_ids)
        self._img_infos_by_img_id: dict['_ImgID', dict[str, Any]] = {
            info['name']: info for infos in self._img_infos_by_cat_id.values() for info in infos}
        self.img_ids: list['_ImgID'] = list(self._img_infos_by_img_id.keys())
        self._cat_id_by_img_id: dict['_ImgID', '_CatID'] = {
            info['name']: cat_id for cat_id, infos in self._img_infos_by_cat_id.items() for info in infos}
        super().__init__(
            imgs_path=imgs_path,
            category_names_by_id={c: c for c in self._cat_id_by_img_id.values()},
            transform=transform, image_loader=image_loader,
            combine_masks=False,
            device=device,
        )

    @classmethod
    def _build_class_to_images(cls, imgs_path: str, allowed_categories: list['_CatID'] = None) -> dict[
        '_CatID', list[dict[str, Any]]]:
        """
        Builds a dictionary mapping class names to lists of image paths.

        Returns:
            dict: {class_name: [list of image paths]}
        """
        class_to_images: dict['_CatID', list[dict[str, Any]]] = defaultdict(list)
        sub_dirs = [d for d in os.listdir(imgs_path) if os.path.isdir(os.path.join(imgs_path, d))]
        for sub_dir in sub_dirs:
            class_dirs = os.listdir(os.path.join(imgs_path, sub_dir))
            for class_dir in class_dirs:

                # filtering
                if allowed_categories is not None and class_dir not in allowed_categories:
                    continue

                # Images within parent dir
                class_to_images |= cls._get_img_info_in_dir(
                    os.path.join(imgs_path, sub_dir, class_dir),
                    class_id=class_dir)

                # Images within subdirs
                subclass_dirs = [item for item in os.listdir(os.path.join(imgs_path, sub_dir, class_dir))
                                 if not os.path.isfile(os.path.join(imgs_path, sub_dir, class_dir, item))]
                for subclass_dir in subclass_dirs:

                    # filtering
                    if allowed_categories is not None and subclass_dir not in allowed_categories:
                        continue

                    class_to_images |= cls._get_img_info_in_dir(
                        os.path.join(imgs_path, sub_dir, class_dir),
                        class_id=class_dir + '+' + subclass_dir)

        return class_to_images

    @staticmethod
    def _get_img_info_in_dir(class_dir_path: str, class_id: str
                             ) -> dict['_CatID', list[dict[str, Any]]]:
        class_to_images: dict['_CatID', list[dict[str, Any]]] = defaultdict(list)
        image_names = [item for item in os.listdir(class_dir_path)
                       if os.path.isfile(os.path.join(class_dir_path, item))]
        for image_name in image_names:
            class_to_images[class_id].append({
                'name': image_name,
                'path': os.path.join(class_dir_path, image_name),
            })
        return class_to_images

    def get_cats(self, img_id) -> list['_CatID']:
        return [self._cat_id_by_img_id[img_id]]

    def get_img_filename(self, img_id: '_ImgID') -> str:
        return os.path.basename(self.get_img_path(img_id))

    def get_img_path(self, img_id: '_ImgID') -> str:
        return self._img_infos_by_img_id[img_id]['path']

    def load_segs(self, img_id: '_ImgID') -> dict[str, bool]:
        cat_ids_in_img = self.get_cats(img_id)
        return {c: (c in cat_ids_in_img) for c in self.cat_ids}


class PlacesDataset(FolderClassificationDataset):
    DEFAULT_IMAGE_LOADER = ImageLoader(img_shape=(256, 256))  # Places dataset only has images of size (256, 256)
    DEFAULT_IMGS_PATH = './data/places_205_kaggle'


class SyntheticBackgroundsDataset(FolderClassificationDataset):
    DEFAULT_IMGS_PATH = './data/synthetic_backgrounds'

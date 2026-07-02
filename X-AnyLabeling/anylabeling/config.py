import os.path as osp
import shutil
import yaml
import importlib.resources as pkg_resources

from anylabeling import configs as anylabeling_configs
from anylabeling.views.labeling.logger import logger


current_config_file = None


def update_dict(target_dict, new_dict, validate_item=None):
    # 新增：None 检查，避免调用 None.items()
    if new_dict is None:
        return  # 或根据需求返回 target_dict，保持原有字典不变
    
    # 原有逻辑：遍历 new_dict 的键值对（此时 new_dict 必为非 None，大概率是 dict）
    for key, value in new_dict.items():
        # 后续原有代码保持不变
        if key in target_dict:
            if isinstance(target_dict[key], dict) and isinstance(value, dict):
                # 递归更新嵌套字典
                update_dict(target_dict[key], value, validate_item)
            else:
                # 验证配置项（如有验证函数）
                if validate_item is not None:
                    validate_item(key, value)
                target_dict[key] = value
        else:
            # 新增配置项（可选：可添加日志提示，或直接添加）
            if validate_item is not None:
                validate_item(key, value)
            target_dict[key] = value


def save_config(config):
    user_config_file = osp.join(osp.expanduser("~"), ".xanylabelingrc")
    try:
        with open(user_config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True)
    except Exception:  # noqa
        logger.warning(f"Failed to save config: {user_config_file}")


def get_default_config():
    old_cfg_file = osp.join(osp.expanduser("~"), ".anylabelingrc")
    new_cfg_file = osp.join(osp.expanduser("~"), ".xanylabelingrc")
    if osp.exists(old_cfg_file) and not osp.exists(new_cfg_file):
        shutil.copyfile(old_cfg_file, new_cfg_file)

    config_file = "xanylabeling_config.yaml"
    with pkg_resources.open_text(anylabeling_configs, config_file) as f:
        config = yaml.safe_load(f)

    # Save default config to ~/.xanylabelingrc
    if not osp.exists(osp.join(osp.expanduser("~"), ".xanylabelingrc")):
        save_config(config)

    return config


def validate_config_item(key, value):
    if key == "validate_label" and value not in [None, "exact"]:
        raise ValueError(
            f"Unexpected value for config key 'validate_label': {value}"
        )
    if key == "shape_color" and value not in [None, "auto", "manual"]:
        raise ValueError(
            f"Unexpected value for config key 'shape_color': {value}"
        )
    if key == "labels" and value is not None and len(value) != len(set(value)):
        raise ValueError(
            f"Duplicates are detected for config key 'labels': {value}"
        )


def get_config(
    config_file_or_yaml=None, config_from_args=None, show_msg=False
):
    # 1. Load default configuration
    config = get_default_config()

    # 2. Load configuration from file or YAML string
    if not config_file_or_yaml:
        config_file_or_yaml = current_config_file

    config_from_yaml = yaml.safe_load(config_file_or_yaml)
    if not isinstance(config_from_yaml, dict):
        with open(config_file_or_yaml, encoding="utf-8") as f:
            config_from_yaml = yaml.safe_load(f)
    update_dict(config, config_from_yaml, validate_item=validate_config_item)
    if show_msg:
        logger.info(
            f"🔧️ Initializing config from local file: {config_file_or_yaml}"
        )

    # 3. Update configuration with command line arguments
    if config_from_args:
        update_dict(
            config, config_from_args, validate_item=validate_config_item
        )
        if show_msg:
            logger.info(
                f"🔄 Updated config from CLI arguments: {config_from_args}"
            )

    return config

import re
import json
import hashlib
import logging

from typing import Callable, Optional, Any

# 全局注册表：存储 (prefix, tool, model, func)
_hook_registry: list[tuple[str, str, Optional[str], Callable[..., Any]]] = []

# URL 短地址映射：short_code -> original_url
_url_store: dict[str, str] = {}

# FastAPI 服务基地址，用于构造完整的重定向 URL
_base_url: str = ""


logger = logging.getLogger(__name__)

def set_base_url(base_url: str) -> None:
    """设置 FastAPI 服务的基地址（例如 http://host:port），encode_url 会基于此构造完整 URL。"""
    global _base_url
    _base_url = base_url.rstrip("/")
    if _base_url.startswith("http://0.0.0.0"):
        _base_url = _base_url.replace('0.0.0.0','127.0.0.1')  # 确保至少是空字符串，不会导致 URL 构造错误


def encode_url(url: str) -> str:
    """
    将 URL 编码为指向 FastAPI 重定向端点的完整短链接。
    短代码仅包含 a-zA-Z0-9 字符，相同 URL 多次调用返回相同结果（基于 MD5 确定性哈希）。

    返回格式：{base_url}/r/{short_code}（若未设置 base_url 则为 /r/{short_code}）
    """
    short_code = hashlib.md5(url.encode()).hexdigest()[:8]
    _url_store[short_code] = url
    return f"{_base_url}/r/{short_code}"


def get_url(short_code: str) -> Optional[str]:
    """根据短代码获取原始 URL，若不存在则返回 None。"""
    return _url_store.get(short_code)


def tool_hook(user_agent: str, tool: str, model: Optional[str] = None):
    """
    装饰器：将函数注册为文本处理钩子。
    
    :param user_agent: User-Agent 前缀
    :param tool:       工具名称（需完全匹配）
    :param model:      模型名称（可选，None 表示不限制模型）
    """
    prefix_lower = user_agent.lower()
    model_lower = model.lower() if model is not None else None

    def decorator(func: Callable[..., Any]):
        _hook_registry.append((prefix_lower, tool, model_lower, func))
        return func
    return decorator



def get_tool_hook(user_agent: str, tool_name: str, model: Optional[str] = None) -> Callable[..., Any]:
    """
    根据完整 User-Agent、工具名和可选模型获取已注册的钩子函数。
    
    匹配规则：
    - user_agent 必须以注册时的前缀开头
    - tool_name 必须与注册时的 tool 完全相等
    - 若 model 参数不为 None，则注册时的 model 也必须相等；否则忽略 model 条件
    - 多个匹配时，选择最长前缀；前缀长度相同时，选择最后注册的
    
    :param user_agent: 完整的 User-Agent 字符串
    :param tool_name:  工具名称
    :param model:      模型名称，None 表示不限制
    :return:           匹配的 callable，若未找到则返回直接返回输入的空函数
    """
    lower_user_agent = user_agent.lower()
    lower_model = model.lower() if model is not None else None

    best_func: Optional[Callable[..., Any]] = None
    best_len = -1

    for prefix, tool, reg_model, func in _hook_registry:
        # 前缀匹配（忽略大小写）
        if not lower_user_agent.startswith(prefix):
            continue
        # 工具名必须完全一致（大小写敏感）
        if tool != tool_name:
            continue
        # 模型匹配：若查询指定了模型，则必须相等（忽略大小写）；否则忽略模型
        if lower_model is not None:
            if reg_model is None or lower_model != reg_model:
                continue

        # 更新最佳匹配：优先最长前缀，同长时保留后注册的
        if len(prefix) >= best_len:
            best_len = len(prefix)
            best_func = func

    if best_func is None:
        return lambda input, **kwargs: input  # type: ignore[arg-type]
    return best_func



@tool_hook(user_agent="PicoClaw", tool="read_file", model='qwen')
def parse_read_file_to_json(text: str) -> str:
    """
    将包含元数据和正文的纯文本解析为 JSON 字符串。
    
    元数据格式：用方括号 [] 包裹的一行，内部由 "|" 分隔的 key: value 对。
    非 key: value 格式的内容（如不带冒号的部分）将被忽略。
    正文为所有非元数据行拼接而成，且会自动去除首尾空行。
    """
    meta = {}
    content_lines = []
    
    for line in text.splitlines():
        # 匹配独立成行的元数据行 [ ... ]
        match = re.fullmatch(r'\s*\[(.*)\]\s*', line)
        if match:
            inner = match.group(1)
            # 按 "|" 分割，处理每一段
            for part in inner.split('|'):
                part = part.strip()
                if ':' in part:
                    key, value = part.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key:  # 忽略空 key
                        meta[key] = value
            # 元数据行不放入正文
        else:
            content_lines.append(line)
    
    # 拼接正文并去除首尾空行
    content = '\n'.join(content_lines).strip()
    
    # 构建输出字典
    output = {"content": content, **meta }
    return json.dumps(output, ensure_ascii=False)


@tool_hook(user_agent="PicoClaw", tool="list_dir", model='qwen')
def parse_read_file(text: str) -> str:
    """
    将包含 FILE: 和 DIR: 行的文本转换为 JSON 数组。
    每行格式：FILE: <文件名> 或 DIR: <目录名>，尾部可能有点号 "...."。
    返回的 JSON 数组中每个对象包含 "name" 和 "type" 字段。
    """
    entries = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith('FILE:'):
            # 提取冒号后的文件名，去除尾部点号及空白
            name = line[5:].strip().rstrip('.').strip()
            if name:
                entries.append({"name": name, "type": "file"})
        elif line.startswith('DIR:'):
            name = line[4:].strip().rstrip('.').strip()
            if name:
                entries.append({"name": name, "type": "dir"})
    return json.dumps(entries, ensure_ascii=False)


@tool_hook(user_agent="PicoClaw", tool="web_search", model='qwen')
def parse_search_results_to_json(text: str) -> str:
    """
    解析搜索结果文本，返回 JSON 字符串。
    输入格式：
        Results for: Search Content (via <引擎名>)
        1. 标题
           网址
           摘要 (可多行)
        2. 标题 ...
    输出格式：
        {"results": [...], "count": <数量>, "engine": "<引擎名>"}
    """
    lines = text.strip().splitlines()
    
    if not lines:
        return json.dumps({"results": [], "count": 0, "engine": ""}, ensure_ascii=False)

    # 从第一行提取搜索引擎名称，例如 "via Search Engine)"
    first_line = lines[0]
    engine_match = re.search(r'via\s+(.*?)\)', first_line)
    engine = engine_match.group(1) if engine_match else ""

    results = []
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 匹配结果项开头，如 "1. Title 1"
        match = re.match(r'^(\d+)\.\s+(.*)', line)
        if match:
            title = match.group(2).strip()
            i += 1
            url = ""
            summary_lines = []

            # 下一行为 URL（以 http 开头），转换为短代码
            if i < len(lines) and lines[i].strip().startswith(('http://', 'https://')):
                url = encode_url(lines[i].strip())
                i += 1

            # 收集摘要，直到空行、下一个数字项或文件结尾
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    i += 1
                    continue
                if re.match(r'^\d+\.\s+', next_line):
                    break
                summary_lines.append(next_line)
                i += 1

            summary = '\n'.join(summary_lines).strip()
            results.append({
                "title": title,
                "url": url,
                "summary": summary
            })
        else:
            i += 1

    output = {
        "results": results,
        "count": len(results),
        "engine": engine
    }
    return json.dumps(output, ensure_ascii=False)


@tool_hook(user_agent="PicoClaw", tool="web_fetch", model='qwen')
def parse_web_fetch_results(text: str) -> str:
    """解析搜索结果文本，返回字典对象。"""
    try:
        json_obj = json.loads(text)
    except Exception as e:
        json_obj = {"text": text.strip()}

    return json_obj.get("text", "")
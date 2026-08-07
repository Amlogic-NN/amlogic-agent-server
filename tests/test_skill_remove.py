import re
from typing import Tuple, List, Dict

def preprocess_system_skills(input: str) -> Tuple[str, str, List[Dict[str, str]]]:
    """
    从 System Prompt 中提取并移除 Skills 相关内容。
    
    返回:
        1. 移除 Skill 相关节后的文本 (str)
        2. Skill 文件路径模板，其中 `{skill_name}` 可用于 str.format (str)
        3. Skill 列表，每个元素为 {"skill_name": name, "description": desc} (list[dict])
    """
    # 1. 提取 Skill 文件路径模板 (Workspace 部分)
    skill_location_template = ""
    # 匹配 ## Workspace 下的 Skills 行，允许 %s 占位符
    workspace_pattern = r'## Workspace.*?- Skills:\s*(.*?)\n'
    match_ws = re.search(workspace_pattern, input, re.DOTALL)
    if match_ws:
        raw_path = match_ws.group(1).strip()
        # 将 {skill-name} 替换为 {skill_name} 以支持 str.format
        skill_location_template = raw_path.replace('{skill-name}', '{skill_name}')
    
    # 2. 查找并移除 # Skills 节（包括标题和后续 <skills>...</skills> 块）
    # 使用非贪婪匹配 .*? 以及 re.DOTALL 让 . 匹配换行
    skills_section_pattern = r'(^|\n)# Skills\b.*?<skills>.*?</skills>'
    match_skills_section = re.search(skills_section_pattern, input, re.DOTALL)
    
    if match_skills_section:
        # 删除整个匹配节，同时注意保留前后的分隔符，避免多余空行
        start, end = match_skills_section.span()
        # 如果匹配开头是 \n，则一起删除，否则删除到结束
        processed_text = input[:start] + input[end:]
        # 处理可能产生的连续空行：将三个以上换行压缩为两个换行（保留段落间距）
        processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)
        # 提取技能列表
        skills_xml = match_skills_section.group(0)
    else:
        # 没有找到 Skills 节，直接使用原文，空列表
        processed_text = input
        skills_xml = ""
    
    # 3. 从 skills_xml 中解析每个 <skill> 的名称和描述
    skills_list: List[Dict[str, str]] = []
    if skills_xml:
        # 找到所有 <skill>...</skill> 块
        skill_blocks = re.finditer(r'<skill>(.*?)</skill>', skills_xml, re.DOTALL)
        for block in skill_blocks:
            skill_content = block.group(1)
            # 提取 name
            name_match = re.search(r'<name>(.*?)</name>', skill_content, re.DOTALL)
            # 提取 description
            desc_match = re.search(r'<description>(.*?)</description>', skill_content, re.DOTALL)
            if name_match and desc_match:
                skill_name = name_match.group(1).strip()
                description = desc_match.group(1).strip()
                skills_list.append({"skill_name": skill_name, "description": description})
    
    return processed_text, skill_location_template, skills_list


if __name__ == "__main__":
    # 示例输入
    import argparse
    parser = argparse.ArgumentParser(description="Test preprocess_system_skills function.")
    parser.add_argument("input_file", help="Path to the input system prompt file.")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        input_text = f.read()

    processed_text, skill_location_template, skills_list = preprocess_system_skills(input_text)

    print("Processed Text:\n", processed_text)
    print("\nSkill Location Template:\n", skill_location_template)
    print("\nExtracted Skills:")
    for skill in skills_list:
        print(f"**{skill['skill_name']}**: {skill['description']}")   
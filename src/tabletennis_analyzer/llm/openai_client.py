from __future__ import annotations

import json
from typing import Any

from openai import OpenAI


def analyze_motion_differences(api_key: str, model: str, summary: dict[str, Any]) -> str:
    if not api_key.strip():
        raise ValueError("API key is required for model analysis.")
    if not model.strip():
        raise ValueError("Model name is required.")

    client = OpenAI(api_key=api_key.strip())
    prompt = _build_prompt(summary)

    response = client.responses.create(
        model=model.strip(),
        input=[
            {
                "role": "system",
                "content": (
                    "You are a table-tennis technique analyst. Analyze per-joint relative speed, "
                    "relative acceleration, and vector-direction differences. Answer in Chinese."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.output_text


def _build_prompt(summary: dict[str, Any]) -> str:
    return (
        "\u8bf7\u6839\u636e\u4e0b\u9762\u7684\u4e52\u4e53\u7403\u52a8\u4f5c\u5dee\u5f02\u6570\u636e\uff0c"
        "\u5206\u6790\u7528\u6237\u76f8\u5bf9\u6559\u7ec3\u6807\u51c6\u52a8\u4f5c\u6700\u9700\u8981\u63d0\u9ad8\u7684\u5730\u65b9\u3002\n"
        "\u91cd\u8981\u8bf4\u660e\uff1a\u8fd9\u4e9b\u901f\u5ea6\u548c\u52a0\u901f\u5ea6\u662f MediaPipe "
        "\u5f52\u4e00\u5316/\u76f8\u5bf9\u5750\u6807\u7a7a\u95f4\u4e0b\u7684\u8fd0\u52a8\u5b66\u91cf\uff0c"
        "\u4e0d\u662f\u7ecf\u76f8\u673a\u6807\u5b9a\u540e\u7684\u771f\u5b9e m/s \u6216 m/s^2\u3002\n"
        "\u6570\u636e\u5df2\u6309\u5f52\u4e00\u5316\u65f6\u95f4\u8f74 time_norm \u5bf9\u9f50\uff0c"
        "diff = user - coach\u3002\n"
        "\u8bf7\u540c\u65f6\u5173\u6ce8\uff1a\n"
        "1. speed_diff \u548c acceleration_magnitude_diff\uff0c\u7528\u4e8e\u5224\u65ad\u5173\u8282\u8fd0\u52a8\u5feb\u6162\u548c\u7206\u53d1\u5dee\u5f02\uff1b\n"
        "2. x/y/z velocity_diff \u548c acceleration_diff\uff0c\u7528\u4e8e\u5224\u65ad\u65b9\u5411\u5206\u91cf\u5dee\u5f02\uff1b\n"
        "3. velocity_angle_abs_diff_deg \u548c acceleration_angle_abs_diff_deg\uff0c\u7528\u4e8e\u5224\u65ad\u8fd0\u52a8\u65b9\u5411\u662f\u5426\u504f\u79bb\u6559\u7ec3\u3002\n"
        "\u8bf7\u8f93\u51fa\uff1a\u6700\u4e3b\u8981\u7684 3-5 \u4e2a\u95ee\u9898\u3001\u6d89\u53ca\u5173\u8282\u3001"
        "\u53ef\u80fd\u6280\u672f\u539f\u56e0\u3001\u53ef\u6267\u884c\u8bad\u7ec3\u5efa\u8bae\u3001"
        "\u9700\u8981\u91cd\u70b9\u89c2\u5bdf\u7684 time_norm \u65f6\u95f4\u6bb5\u3002\n\n"
        f"{json.dumps(summary, ensure_ascii=False)}"
    )

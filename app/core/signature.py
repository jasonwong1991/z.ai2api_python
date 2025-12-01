#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Z.AI API 签名功能
"""

import time
import hmac
import hashlib
import json
import base64
from typing import Dict, Any
from app.utils.helpers import debug_log


def decode_jwt_payload(token: str) -> Dict[str, Any]:
    """
    解码JWT token的payload部分
    
    Args:
        token: JWT token字符串
        
    Returns:
        解码后的payload字典
    """
    parts = token.split('.')
    payload = parts[1]

    padding = 4 - len(payload) % 4
    if padding != 4:
        payload += '=' * padding

    decoded = base64.urlsafe_b64decode(payload)
    return json.loads(decoded)


def zs(e: str, t: str, timestamp: int) -> Dict[str, str]:
    """
    生成Z.AI API签名 (匹配最新的 JavaScript zs 函数)

    Args:
        e: 签名元数据字符串，格式为 "requestId,{requestId},timestamp,{timestamp},user_id,{user_id}"
        t: 最近一次user content (用于签名的提示词)
        timestamp: 时间戳（毫秒）

    Returns:
        包含签名和时间戳的字典
    """
    # r = Number(s) - 时间戳数值
    r = timestamp
    # i = s - 时间戳字符串
    i = str(timestamp)

    # a = n.encode(t) - UTF-8 编码用户内容
    a = t.encode('utf-8')

    # w = btoa(String.fromCharCode(...a)) - Base64 编码
    w = base64.b64encode(a).decode('ascii')

    # c = `${e}|${w}|${i}` - 构建签名字符串
    c = f"{e}|{w}|{i}"

    # E = Math.floor(r / (5 * 60 * 1e3)) - 计算时间窗口
    E = r // (5 * 60 * 1000)

    # A = CryptoJS.HmacSHA256(`${E}`, "key-@@@@)))()((9))-xxxx&&&%%%%%")
    secret = "key-@@@@)))()((9))-xxxx&&&%%%%%"
    A = hmac.new(secret.encode('utf-8'), str(E).encode('utf-8'), hashlib.sha256).hexdigest()

    # k = CryptoJS.HmacSHA256(c, A).toString()
    k = hmac.new(A.encode('utf-8'), c.encode('utf-8'), hashlib.sha256).hexdigest()

    return {
        "signature": k,
        "timestamp": i
    }


def generate_zs_signature(token: str, request_id: str, timestamp: int, user_content: str) -> Dict[str, str]:
    """
    生成Z.AI API签名的便捷函数
    
    Args:
        token: JWT token
        request_id: 请求ID
        timestamp: 时间戳（毫秒）
        user_content: 最近一次user content
        
    Returns:
        包含签名和时间戳的字典
    """
    # 从token中提取user_id
    try:
        payload = decode_jwt_payload(token)
        user_id = payload['id']
    except Exception as e:
        debug_log(f"解码JWT token失败: {e}")
        user_id = "guest-user"
    
    # 构建签名字符串
    e = f"requestId,{request_id},timestamp,{timestamp},user_id,{user_id}"
    
    # 调用zs函数生成签名
    return zs(e, user_content, timestamp)
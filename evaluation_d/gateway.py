"""保留 v1 的禁用网关供对照；v2 学校实现位于 school_gateway.py。"""


class DisabledGateway:
    """暂未启用的模型网关。接入正式实现时保持相同调用入口。"""

    def call_llm(self, *args, **kwargs):
        raise NotImplementedError("真实模型网关尚未启用；当前不会发送请求或记录虚构 usage。")


def call_llm(*args, **kwargs):
    """占位函数，后续可由符合文档接口的正式网关替换。"""
    raise NotImplementedError("真实模型网关尚未启用；当前不会发送请求或记录虚构 usage。")

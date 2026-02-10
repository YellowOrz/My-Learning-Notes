# 工具

## CLI

### Claude Code

- 安装：[官方的部署脚本](https://code.claude.com/docs/en/setup#installation)、[硅基流动的部署脚本](https://docs.siliconflow.cn/cn/usercases/use-siliconcloud-in-ClaudeCode#%E6%96%B9%E5%BC%8F%E4%B8%80%EF%BC%9A%E4%B8%80%E9%94%AE%E5%AE%89%E8%A3%85%E5%8F%8A%E9%85%8D%E7%BD%AE%E8%84%9A%E6%9C%AC)

    - 如果启动claude报如下错，在`~/.claude.json`中添加内容`"hasCompletedOnboarding": true`

        ```
        Unable to connect to Anthropic services
        Failed to connect to api.anthropic.com: ERR BAD REQUEST
        lease check your internet connection and network settings.
        Note: Claude Code might not be available in your country, Check supported countries atnttps://anthropic.com/supported-countriesS E:ltoollclaude code>
        ```

- 设置第三方的多个模型：以bash为例

    ```bash
    export ANTHROPIC_BASE_URL="https://api.siliconflow.cn/"
    export ANTHROPIC_API_KEY="sk-xxx"
    export ANTHROPIC_MODEL="Pro/deepseek-ai/DeepSeek-V3.2"	# 得有一个默认的模型
    _claude() {
        echo "使用$1, 价格 ¥$2/ M Tokens"
        export ANTHROPIC_MODEL="$1"
        claude
    }
    claude_kimi(){
        name="Pro/moonshotai/Kimi-K2.5"
        price=21
        _claude $name $price
    }
    claude_glm(){
        name="Pro/zai-org/GLM-4.7"
        price=16
        _claude $name $price
    }
    claude_deepseek(){
        name="Pro/deepseek-ai/DeepSeek-V3.2"
        price=3
        _claude $name $price
    }
    claude_minmax(){
        name="Pro/MiniMaxAI/MiniMax-M2.1"
        price=8.4
        _claude $name $price
    }
    ```
    
    

# 测评

- [Arena Leaderboard](https://arena.ai/zh/leaderboard)：类别包含Text、Code、Vision、Text-to-Image、Image、Edit、Search、Text-to-Video、Image-to-Video
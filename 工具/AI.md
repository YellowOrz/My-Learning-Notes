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

# 教程

## Learn Claude Code by doing, not reading

>  https://claude.nagdy.me/

### Setting Up Claude Code for a Project

- CLAUDE.md 文件不超过 200 行。每一行都应与几乎所有会话相关
  - 如果某内容仅针对某一项功能，应将其放入路径专属的规则文件中\
  - 最有价值的部分包括：技术栈和版本、开发命令（安装、测试、构建、代码检查）、不常见的命名规范，以及会让新开发者踩坑的已知潜在问题。

### Commands in Depth

- claude code的内置技能
  - `/simplify`审阅最近修改的文件以评估代码质量，同时生成并行的审阅智能体，负责不同方面的检查。
  - `/batch <instruction>` 适用于跨多个文件的大规模修改——它会规划工作、使用独立的 git 工作树，还能协调验证工作以及面向拉取请求（PR）的后续处理。
  - `/loop 5m check deploy status` 会按固定间隔重复执行指令，适用于轮询长时间运行的操作。
    - `/proactive` 是 `/loop` 的别名
- 快捷键
  | 快捷键        | 说明                                                         |
  | ------------- | ------------------------------------------------------------ |
  | Ctrl+O        | 进入详细模式，实时查看工具调用和推理步骤                     |
  | Ctrl+B        | 让正在运行的 bash 命令和代理在后台继续执行，从而在它们继续工作的同时给 Claude 下达另一条指令 |
  | Ctrl+X Ctrl+K | 终止所有后台代理                                             |
  | Ctrl+U        | 清空整个输入缓冲区                                           |
  | Ctrl+Y        | 恢复你刚刚清空的内容                                         |
  | Ctrl+L        | 除了清除提示输入外，还会强制全屏重绘，可用于终端输出出现撕裂或偏移的情况 |

### Agent Skills

- Plugin skills 使用 `plugin-name:skill-name` 命名空间
- Skills分3个层级进行加载
- 层级一：YAML格式的信息，用来描述技能，永远都会加载，以便 Claude 了解可用的功能。包含的字段如下
  - `name`：skill的名字，在claude中可以通过`/name`手动调用该技能
  - `description`：最重要的部分，决定了 Claude 何时自动调用该技能，必须精准描述，比如使用任务类型（“scan”, “generate”, “analyze”）、主题领域（“security”, “API”, “database”）以及明确的触发短语（“when the user mentions”, “use when”）。类似 “helps with code”这样模糊的描述永远不会触发调用
  - `when_to_use`：大致上是`description`的扩展。`description`和`when_to_use`文本合并后截断为1536个字符后记录到skill列表中，剩下的再记录到`when_to_use`
    - Claude 将skill描述的总空间预算设定为上下文窗口的约 1%，必要时提供 8000 个字符的备用容量，而设置中 `SLASH_COMMAND_TOOL_CHAR_BUDGET` 可以提高这一上限
    - 运行 `/context` 可检查skill是否未被列入列表
  - `shell `：指定用于 `!command`块的命令行解释器。比如在Windows上使用powershell（或者设置`CLAUDE_CODE_USE_POWERSHELL_TOOL=1`）
  - `disable-model-invocation: true` 表示只有用户能通过 /skill-name 调用它，Claude 永远不会自动触发——用于任何带有副作用的技能（部署、推送、发送操作）
  - `user-invocable: false` 会将技能从 / 菜单中隐藏，但仍允许 Claude 自动调用它——适合那些无法作为命令执行的背景知识类技能
  - `paths`：接受一个 YAML 格式的通配符列表，用于限定技能的生效范围
  - `context: fork` 在一个具有独立上下文窗口的隔离子智能体中运行该技能。
  - `agent`：指定智能体类型，Explore用于只读研究，Plan用于规划，general-purpose适用于需要所有工具的任何任务。
    - 子智能体承担繁重工作的同时，主对话保持简洁。
  - `argument-hint`：显示技能期望的参数
  - `allowed-tools`：限制技能运行时可使用的工具，其遵循与权限规则相同的模式语法
- 层级二：SKILL.md的全文（推荐长度在500行以内），只有在Claude需要使用当前skill的时候加载
  - 使用`!command`的语法会在技能内容发送给 Claude 之前执行 shell 命令。输出会被内联处理——Claude 只能看到结果，无法看到命令。
  - 捕获参数的方式有2种：`$ARGUMENTS` 会捕获命令名之后的所有内容作为单个字符串。`$0、$1、$2` 会捕获以空格分隔的单个参数
- 层级三：在skill目录中的辅助文件（模板、脚本），通过bash按需加载
  - 辅助文件通过相对路径进行引用
  - 将 SKILL.md 的内容控制在 500 行以内；将详细的参考资料放在单独的辅助文件中
- claude code的内置技能
  - `/less-permission-prompts`（v2.1.112 版本新增功能）会扫描你的对话记录，查找常见的只读 Bash 和 MCP 工具调用，然后为你的 .claude/settings.json 生成一份优先许可白名单。在使用几次会话后运行该功能，就能生成适配你实际工作流程的权限配置


-  MCP tools follow the pattern mcp__servername__toolname, so mcp__github__.* 
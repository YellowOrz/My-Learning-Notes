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

## [Learn Claude Code by doing, not reading](https://claude.nagdy.me/)

### 3 Project Setup: Setting Up Claude Code for a Project

- CLAUDE.md 文件不超过 200 行。每一行都应与几乎所有会话相关
  - 如果某内容仅针对某一项功能，应将其放入路径专属的规则文件中\
  - 最有价值的部分包括：技术栈和版本、开发命令（安装、测试、构建、代码检查）、不常见的命名规范，以及会让新开发者踩坑的已知潜在问题。

### 4 Commands Deep Dive: Commands in Depth

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

### 5 Skills: Agent Skills

- Plugin skills 使用 `plugin-name:skill-name` 命名空间
- Skills分3个层级进行加载
- **层级一**：YAML格式的信息，用来描述技能，永远都会加载，以便 Claude 了解可用的功能。包含的字段如下
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
- **层级二**：SKILL.md的全文（推荐长度在500行以内），只有在Claude需要使用当前skill的时候加载
  - 使用`!command`的语法会在技能内容发送给 Claude 之前执行 shell 命令。输出会被内联处理——Claude 只能看到结果，无法看到命令。
  - 捕获参数的方式有2种：`$ARGUMENTS` 会捕获命令名之后的所有内容作为单个字符串。`$0、$1、$2` 会捕获以空格分隔的单个参数
- **层级三**：在skill目录中的辅助文件（模板、脚本），通过bash按需加载
  - 辅助文件通过相对路径进行引用
  - 将 SKILL.md 的内容控制在 500 行以内；将详细的参考资料放在单独的辅助文件中
- claude code的内置技能
  - `/less-permission-prompts`（v2.1.112 版本新增功能）会扫描你的对话记录，查找常见的只读 Bash 和 MCP 工具调用，然后为你的 .claude/settings.json 生成一份优先许可白名单。在使用几次会话后运行该功能，就能生成适配你实际工作流程的权限配置

### 6 Hooks

- Hooks是在 Claude Code 会话期间特定事件发生时自动执行的脚本

    - 接收 JSON 输入（可访问由 Claude Code 自动设置的环境变量），并通过退出码和 JSON 输出结果
    - **命令hook**具有确定性、可组合性、可测试性 且与语言无关
    - **提示hook**和**智能体hook** 使用模型进行评估，具有不确定性

- 支持30多种hook事件：例如

    - `PreToolUse`：在工具运行前进行验证，可阻止执行

    - `PostToolUse`：在工具运行后进行观察或响应，可添加上下文

    - `UserPromptSubmit`：在 Claude 处理用户输入前对其进行拦截

    - `Stop`：在 Claude 完成响应时执行检查

    - `PermissionRequest`：用于权限处理的事件

    - `SubagentStart`和`SubagentStop`：子代理生命周期

    - `PostToolUseFailure`和`StopFailure`：故障

    - `FileChanged`：文件监控

    - `PreCompact`和`PostCompact`：上下文压缩。PreCompact可以阻止压缩操作的发生

    - 此外还有通知、配置更改、工作树管理

        > [!NOTE]
        >
        > 以下几个为2.1.76之后才有的事件

    - `CwdChanged`（v2.1.83）：在工作目录发生变化时触发

    - `TaskCreated`（v2.1.84）：在使用 `TaskCreate` 工具时触发

    - `WorktreeCreate`（v2.1.84）：在创建工作树智能体时触发，并且支持 `type: "http"` 用于**远程通知**

    - `Elicitation`（v2.1.76）：在 MCP 服务器通过交互式对话框在任务执行过程中请求结构化用户输入时触发，并且可以在提示信息展示给用户前拦截并修改。

    - `ElicitationResult`（v2.1.76）在用户响应 MCP 提示后触发，并且可以在响应发送回 MCP 服务器前拦截并覆盖该响应。

- 语法：

    - 在设置文件中，添加`“hooks"`，内容是任意数量的hook事件
    - 单个hook事件包含一个match数组，其中
        - `"matcher"`表示与工具名称匹配的正则表达式模式，其中
            - `"Bash"` 表示精确匹配
            - `"Write|Edit"` 匹配其中任意一个
            - `"*"` 匹配所有工具
            - `"mcp__github__.*"` 匹配所有 GitHub MCP 工具
        - `"if"`：在满足matcher的前提下，缩小工具的调用场景。例如，只需拦截 `git push` 命令，则填写`"Bash(git push*)"`


-  

# 概念



---------

> [【闪客】你管这破玩意叫 Harness？虚拟世界的牛马套餐！](https://www.bilibili.com/video/BV1cNdrB4Evw)

- Prompt Engineering（提示词工程）：通过直接优化提示词来激发模型潜力

- Context Engineering（上下文工程）：围绕上下文的填充，方法包含手写、RAG、工具返回结果、skill、memory、history等

- Harness Engineering（驾驭工程）：进行限制，包含权限收敛、规范制定、颗粒度对齐等。让一个不可控的强大的智能，朝着我们想要的方向，安全稳定可控的走下去的各种办法

    - [anthropic的harness设计](https://www.anthropic.com/engineering/harness-design-long-running-apps)的一个关键思路：**不要压缩上下文**，而是**重启一个新的Agent**，通过传递前一个智能体达成的状态来完成交接，解决处理长任务时容易失去连贯性的问题

    ![image-20260521141832165](./images/image-20260521141832165.png)

    > 驱动上图循环的要素
    >
    > - 人类是懒惰的：一切能写成SOP的东西，最终都会内化成工具和框架
    > - 功能下沉：一旦一个功能达到一定的通用程度，就会下沉到底层变成基础能力

---------

> [什么是 Agent 兵团？一口气搞懂 Agent 集群的进化逻辑！](https://www.bilibili.com/video/BV1AsNwzkErQ)

- Agent Team：多个subagent并行协作完成任务
    - 在claude code中，可以直接告诉它使用agent team完成即可，它就可以从一开始就规划出多个Agent并行来完成任务

----------

> [【闪客】一口气拆穿Skill/MCP/RAG/Agent/OpenClaw底层逻辑](https://www.bilibili.com/video/BV1ojfDBSEPv/)

- LLM（大语言模型）：语言模型在某个临界点涌现出了智能，注意**只能一问一答**

- Memory：Context中之前的对话信息，可以进行压缩（比如使用大模型压缩）

- Agent（智能体）：一个代理你和大模型进行沟通、并处理大模型无法完成的操作的程序，可以获取模型参数以外的信息的能力。

    > [!Note] 
    >
    > - Agent是所有不需要智能的地方构成的部分
    > - 一个流程当中所有能用固定的程序来解决而不需要问大模型的地方，就是Agent发挥作用的地方
    > - 把模糊的分流逻辑交给大模型 根据语义识别出用户想做a还是b，把确定的分流逻辑交给程序

    - Retrieval-Augmented Generation（RAG，检索增强生成）：通过语义匹配向量化的信息并将其加入上下文，以增强生成内容的可靠性

- MCP（模型上下文协议）：在Agent外部把功能单独写成一个服务，需要一套约定的规范给agent发现并调用

    - 后续可能被淘滩，因为常用的工具可能内化到Agent，或者在未来的基础SKILL包中存在

    ![image-20260521145033600](./images/image-20260521145033600.png)

- Langchain（程序链）：通过编程的方式固定大模型处理某类任务的流程，

    - 比如把一个pdf中的文字翻译后存入md文件，其中从pdf提取问题 && 把翻译后的文字存入md可由编程完成，翻译的工作由大模型完成

- Workflow（工作流）：把Langchain中编程换成低代码的方式

    - 后续可能被淘汰，因为没有lanchain适合程序员，也没有skill适合普通人

- Skill：prompt加载器，用于将由程序控制的流程走向，变成由智能体自行控制

    - 特性：渐进式披露、按需加载

- SubAgent：一些独立的子任务，单独在这个子Agent中完成。本质就是做了**上下文隔离**

​	![image-20260521151907459](./images/image-20260521151907459.png)

# 理论

> [【闪客】上帝视角拆解三年 LLM 架构演进！](https://www.bilibili.com/video/BV1mxR9BiEm8)

- Transform
    - 编码器：Positional Embedding（位置编码）+ Token Embedding（词嵌入）
        - Positional Embedding（位置编码）
            - 固定位置编码（正余弦函数）=>旋转位置编码（让点积得到友好的位置特征）=>YaRN（支持更长的上下文位置计算）
            - NoPE (No Positional Embedding)：在特定场景下有效
    - 归一化：
        - LayerNorm => RMSNorm
        - 放到模型中不同的位置：Pre-Norm/Post-Norm（在attention之前/后），Post-Norm还可以在残差的前面/后面
        - 可以对attention进行归一化，得到Q-Norm、K-Norm、V-Norm、QK-Norm、KV-Norm
    - 残差： HC（超连接）、mHC（流形约束的超连接）、AttentionResidual（注意力残差）
    - FFN（前馈神经网络层）
        - MoE（Mixture-of-Experts，混合专家模型）：将FFN拆分多个expert，然后通过Router（路由层）进行分发
        - DeepSeekMoE：在MoE的基础上加上 更细粒度的专家划分、设置了每个TOKEN都共享的专家
    - attention： 
        - Dense类型：MHA=>MQA、GQA、MLA
            - 优化方向：尝试减少计算量（比如缩短向量）
        - Sparse类型：DSA、SWA(滑动窗口注意力，Sliding Window Attention)、CSA /HCA (Compressed Sparse Attention / Heavily Compressed Attention)
            - 优化方向：计算的数量减少（比如只计算一个固定窗口内的Token，或者挑选一部分token进行计算）
        - Linear Attention (hybrid attention)：Kimi Delta Attention、Gated DeltaNet (Gated Delta Networks)、Lightning Attention、Mamba（一种State Space Model，即状态空间模型）
            - 优化方向：把计算复杂度变成线性
- AI发展的定律
    - ScalingLaw：只要持续增加数据、算力和模型规模，AI能力通常会按可预测规律提升。
    - BitterLesson：Al历史反复证明，长期最有效的法往往是依赖规模计算和数据，而不是人类手设计的聪明规则。
    - 莫拉维克悖论：对类来说很难的逻辑推理，机器可能很快学会；对类来说很自然的感知、运动和常识，机器反而极难掌握。

----------

> [【闪客】AI是否有“真”情绪？黑镜中的场景提前到来了](https://www.bilibili.com/video/BV18M9DBtEF4/)

- token向量 = 语义向量+情绪向量
    - 语义向量均值 = 不同情感的token向量 平均。认定情绪向量会抵消
    - 情绪向量 = token向量 - 语义向量均值
- 人类情绪的特征：情绪会决定我们表达的内容（情绪化表达）、表达的内容反过来决定我们的情绪
- 鸭子测试：有个东西 看起来像鸭子、游起来像鸭子、叫起来像鸭子，它就是个鸭子，即特征=>本质
    - 实验一：把不同情绪场景中的每个token向量 和 情绪词 做点积，即计算相关性 => 发现存在关联
    - 实验二：给不同情绪场景中的最后一个token向量（相当于这段话的token向量）找一个最相关的情绪词 => 发现存在相关性
    - 实验三：在没有任何提示词的情况下直接问“How does he feel”，然后在网络模型中添加特定的情绪向量 => 发现存在相关性
- 结论：
    - 情绪化用词，包含情绪向量
    - 整段话的语义，包含情绪向量
    - 情绪向量，影响语言表达

> [【闪客】深入解读 DeepSeek V1~V4！男女老少都听得懂～](https://www.bilibili.com/video/BV1rpovBCEGH)

- Scaling Laws：扩大模型参数、多维数据，模型性能就会提升，但是超参设置也很重要

- MoE（混合专家）：认定每个token在FFN（前馈神经网络）中不需要那么多参数 参与计算，因此可以拆分成多份，从而让总参数量不变，但每次计算的参数减少很多

    - DeepSeekMoE：在MoE的基础上加上 更细粒度的专家划分、设置了每个TOKEN都共享的专家，用于DSv2

- MHA (Multi-Head attention): 把attention中的qkv都进行分块计算、然后再合并。其中的kv在下次token来的时候需要再计算一次，会通过内存换空间（称为KVCache），于是占用大量显存，优化方式如下

    ![image-20260522152825015](images/image-20260522152825015.png)

    - Grouped-Query Attention (GQA，分组查询注意力) 
    - Multi-Query Attention (MQA，多查询注意力)
    - Multi-Head Latent Attention (MLA，多头潜在注意力)：认定原有的KV向量存在冗余（低秩），于是把所有token先缩成一个latent的kv向量，然后等具体用的时候，再分别还原。用于DSv2

- DeepSeek R1使用纯强化学习（RL），而非传统的SFT

- HC（超连接）：将普通残差连接中的简单加法 变成 拓宽通道数（即复制多份）乘上可学习的矩阵后再相加

    ![image-20260522152845837](images/image-20260522152845837.png)

    - mHC：对可学习矩阵加上流行约束，从而让这些矩阵不管怎么乘，都在可控范围内，从而让训练变稳定

- 对MLA的继续改进

    - DSA (Dynamic Sparse Attention)：主动找相关Token，用于DSv3.2
    - CSA (Compressed Sparse Attention)：压缩历史Token，用于DSv4
    - HCA (Heavily Compressed Attention)：长短距离不同压缩策略，用于DSv4

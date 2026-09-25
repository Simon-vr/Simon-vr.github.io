# simon-vr

对计算机体系结构感兴趣，喜欢从零实现一些底层系统，也记录学习和生活。

网站：<https://simon-vr.github.io/>

## 项目

- **YScore** — 从零手写的 RV32 SoC：Verilog 实现的 5 级多周期 RISC-V 处理器（RV32I + Zicsr）、AXI4-Lite 总线、UART/GPIO/CLINT 外设，并在其上运行裸机实时系统（RTOS）。先在 QEMU 验证，再移植到 Cyclone IV E FPGA。
  → <https://simon-vr.github.io/YScore/>

- **FPGA-Router-Experiment** — 从零实现的 C++17 FPGA 布线引擎：用 Kruskal 最小生成树拆分多端线网，以 BFS / A* / Mikami-Tabuchi 完成详细布线，并用 OpenMP 并行的 PathFinder 协商布线化解拥塞，附带交互式可视化。
  → <https://simon-vr.github.io/FPGA-Router-Experiment/>

## 博客

主要记录几类内容：

- **学习资料**：数学分析、线性代数、C/C++、数字电路、物理、离散数学、数据结构等课程的笔记与真题。
- **读书与观影**：一些书评和电影随笔。
- **想法与考据**：日常思考、历史与文本方面的小考据。

## 联系

- 邮箱：simony@tutamail.com
- 也欢迎在文章下方评论区留言。

---

## English

Interested in computer architecture, and enjoys building low-level systems from scratch. Also writes about study notes and daily life.

Website: <https://simon-vr.github.io/>

**Projects**

- **YScore** — an RV32 SoC written from scratch: a Verilog 5-stage multi-cycle RISC-V core (RV32I + Zicsr), an AXI4-Lite bus, UART/GPIO/CLINT peripherals, running a bare-metal real-time system, verified on QEMU then ported to a Cyclone IV E FPGA.
- **FPGA-Router-Experiment** — a from-scratch C++17 FPGA routing engine: Kruskal MST net decomposition, BFS / A* / Mikami-Tabuchi detail routing, and an OpenMP parallel PathFinder negotiated router with interactive visualization.

**Blog** — study materials (math analysis, linear algebra, C/C++, digital circuits, physics, discrete math, data structures), reading and film notes, and occasional thoughts.

**Contact** — simony@tutamail.com

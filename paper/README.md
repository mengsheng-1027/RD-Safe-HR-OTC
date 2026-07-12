
# OTC 中文课程论文 XeLaTeX 初稿

## 编译方式

在 VSCode 中安装 LaTeX Workshop 后，使用 XeLaTeX 编译 `main.tex`。

推荐命令：

```cmd
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

如果暂时不想生成参考文献，也可以只运行：

```cmd
xelatex main.tex
```

## 文件结构

```text
main.tex
references.bib
figures/
tables/
```

## 后续需要替换的占位内容

1. `fig:framework`：RD-Safe+HR-OTC 总体框架图。
2. `tab:negative_transfer`：Target-only vs Original OTC-GW 负迁移表。
3. `tab:main_results`：三 seed 主结果表。
4. `fig:main_bar`：主实验柱状图。
5. `tab:component_ablation`：主模块消融。
6. `tab:hr_ablation`：HR 三信号消融。
7. `tab:strict`：strict unseen-pair 鲁棒性表。
8. `tab:gate`：门控统计表。

## 注意

当前版本刻意用 `\tabplaceholder` 和 `\figplaceholder` 保证没有图片表格时也能编译。
等结果表和图片最终确认后，把占位宏替换成正式 `tabular` 或 `\includegraphics`。

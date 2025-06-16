from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r"""\usepackage{cmbright}
\usepackage{amsfonts}
\DeclareFontShape{OT1}{cmss}{m}{it}{<->ssub*cmss/m/sl}{}
\renewcommand{\rmdefault}{cmss}
\renewcommand{\sfdefault}{cmss}"""
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'cm'
plt.rcParams['font.size'] = 10
plt.rcParams['svg.fonttype'] = 'none'
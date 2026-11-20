"""Export a snapshot of the recovery training curves (requires matplotlib)."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    raw=(a.run/'log.csv').read_text()
    rows=list(csv.DictReader(io.StringIO(raw)))
    rows=[r for r in rows if r.get('recovery_level') is not None]
    col=lambda key: np.array([float(r[key]) for r in rows])
    x=col('iter');last=int(x[-1])
    args_path=a.run/'args.json'
    total=int(json.loads(args_path.read_text()).get('iters',last)) if args_path.exists() else last
    full_difficulty=int(.7*total)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if Path(font).exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.titlesize':13,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8.8))
    fig.patch.set_facecolor('#f5f7fb')
    fig.suptitle(f'D405 RGB-D 救球训练 · 第 {last} / {total} 次',fontsize=20,x=.06,ha='left',y=.97,weight='bold')
    fig.text(.06,.925,'训练回合统计，不是独立测试结果；曲线使用日志中的近期已结束回合均值（最多 200 回合）。',fontsize=11,color='#526174')
    for ax in axes.flat:
        ax.set_facecolor('white')
        ax.grid(axis='y',color='#e8edf4')
        ax.set_axisbelow(True)
        ax.set_xlim(0,max(total,last))
        ax.set_xlabel('训练迭代次数')
        ax.spines['left'].set_color('#cbd5e1');ax.spines['bottom'].set_color('#cbd5e1')
        ax.axvline(full_difficulty,color='#94a3b8',ls=':',lw=1)
    ax=axes[0,0]
    success=col('success')*100
    ax.plot(x,success,color='#1463dc',lw=2,label='严格成功率')
    ax.plot(x,col('recovery_level')*100,color='#d58a20',ls='--',lw=1.5,label='扰动难度（上限=100%）')
    ax.set_ylim(0,105);ax.set_ylabel('%')
    ax.set_title(f'成功率与扰动难度   ·   最新 {success[-1]:.1f}%',loc='left')
    ax.legend(loc='upper left',fontsize=9,frameon=False)
    ax=axes[0,1];ax.plot(x,col('ep_return'),color='#2463a6',lw=1.8)
    ax.set_title(f"单回合奖励   ·   最新 {col('ep_return')[-1]:.1f}",loc='left')
    ax.set_ylabel('回合总奖励')
    ax=axes[1,0];height=col('height_error_m')*1000
    ax.plot(x,height,color='#148777',lw=1.8);ax.axhline(0,color='#64748b',lw=1,ls='--')
    ax.set_title(f'终局拍高有符号平均偏差   ·   最新 {height[-1]:+.2f} mm',loc='left')
    ax.set_ylabel('拍心高度 − 目标高度（mm）')
    ax=axes[1,1];energy=col('energy_j')
    ax.plot(x,energy,color='#9d51ac',lw=1.8)
    ax.set_title(f'单回合绝对机械功   ·   最新 {energy[-1]:.2f} J',loc='left')
    ax.set_ylabel('∫ Σ|关节力矩 × 角速度| dt（J）')
    fig.text(.06,.057,f'竖虚线：第 {full_difficulty} 次起达到本次配置的最高扰动难度。高度正负误差可能抵消，不代表每回合达标。',fontsize=10,color='#526174')
    fig.text(.06,.032,'各回合难度和持续时间不同，能量下降不能直接证明更省电；应以同条件独立测试比较。',fontsize=10,color='#526174')
    fig.subplots_adjust(left=.08,right=.97,top=.865,bottom=.14,hspace=.38,wspace=.24)
    out=a.run/f'training_curves_iter_{last}.png'
    fig.savefig(out,dpi=150,facecolor=fig.get_facecolor())
    (a.run/f'training_curve_data_iter_{last}.csv').write_text(raw)
    print(out.resolve())


if __name__=='__main__':
    main()

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys, os
OUT = sys.argv[1]
NAVY="#0B2545"; DEEP="#13315C"; MID="#1F6FB2"; SKY="#4EA8DE"; LIGHT="#A9D6F5"; PALE="#E7F1FB"; GREY="#8A94A6"; INK="#1B2533"
plt.rcParams.update({"font.family":"Carlito","font.size":11,"axes.edgecolor":"#C9D3E0","axes.labelcolor":INK,
  "xtick.color":INK,"ytick.color":INK,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold",
  "axes.titlesize":13,"axes.titlecolor":NAVY,"axes.titlelocation":"left","axes.titlepad":10})
def save(fig,name):
    fig.savefig(os.path.join(OUT,name),dpi=220,bbox_inches="tight",facecolor="white"); plt.close(fig)

def hbar(name,title,labels,vals,fmt,colors,xlabel=None,xlim=None,figsize=(7.2,3.0),note=None):
    fig,ax=plt.subplots(figsize=figsize)
    y=range(len(labels))
    ax.barh(list(y),vals,color=colors,height=0.62)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y",length=0)
    ax.xaxis.grid(True,color="#E3E9F2",lw=0.8); ax.set_axisbelow(True)
    m=max(vals)
    for i,v in enumerate(vals):
        ax.text(v+m*0.015 if v>=0 else v-m*0.015,i,fmt(v),va="center",ha="left" if v>=0 else "right",color=INK,fontsize=11,fontweight="bold")
    if xlim: ax.set_xlim(*xlim)
    else: ax.set_xlim(0,m*1.15)
    if xlabel: ax.set_xlabel(xlabel)
    ax.set_title(title)
    if note: fig.text(0.01,-0.02,note,fontsize=9,color=GREY,ha="left",va="top")
    save(fig,name)

# 1 architecture
fig,ax=plt.subplots(figsize=(8.6,3.6)); ax.set_xlim(0,10); ax.set_ylim(0,4.6); ax.axis("off")
def box(x,y,w,h,t,s,fc,tc="white"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.12",fc=fc,ec="none"))
    ax.text(x+w/2,y+h*0.64,t,ha="center",va="center",color=tc,fontsize=11.5,fontweight="bold")
    ax.text(x+w/2,y+h*0.3,s,ha="center",va="center",color=tc,fontsize=9.2)
def arr(x1,y1,x2,y2,c=MID):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,color=c,lw=1.6))
top=[("Host port","SPI / QSPI",PALE,NAVY),("SRAM","4 × 1 KB",PALE,NAVY),("DPU array","P × 16 MACs",MID,"white"),
     ("Softmax","base-2 online",DEEP,"white"),("Accumulate","P·V shift-add",MID,"white"),("Output","× 1/l, MXINT8",PALE,NAVY)]
w=1.45; g=0.18
for i,(t,s,fc,tc) in enumerate(top):
    x=0.1+i*(w+g); box(x,2.55,w,1.25,t,s,fc,tc)
    if i<5: arr(x+w+0.01,3.17,x+w+g-0.01,3.17)
ax.text(0.1,4.25,"Streaming compute path",color=NAVY,fontsize=12,fontweight="bold")
ax.add_patch(FancyBboxPatch((3.30,2.42),4.95,1.52,boxstyle="round,pad=0.02,rounding_size=0.15",fc="none",ec=SKY,lw=1.4,ls=(0,(4,3))))
side=[("RoPE rotator","rotates K on write"),("Heavy-hitter tracker","picks the entry to evict"),("Controller","one FSM, 3 dataflows"),("Chip top","clock gate, reset sync")]
w2=2.3
for i,(t,s) in enumerate(side):
    x=0.1+i*(w2+0.17); box(x,0.35,w2,1.15,t,s,"#DCE9F7",NAVY)
ax.text(0.1,1.78,"LLM support and control",color=NAVY,fontsize=12,fontweight="bold")
save(fig,"fig_arch.png")

# 2 online softmax steps
fig,ax=plt.subplots(figsize=(8.6,1.9)); ax.set_xlim(0,10); ax.set_ylim(0,2); ax.axis("off")
steps=[("1  Score","s = q · k"),("2  Max","integer running max"),("3  Weight","p = 2^(s − m), LUT"),("4  Rescale","shift l and acc"),("5  Finish","one 1/l per row")]
cols=[PALE,LIGHT,SKY,MID,NAVY]
for i,(t,s) in enumerate(steps):
    x=0.05+i*1.99
    ax.add_patch(FancyBboxPatch((x,0.35),1.85,1.3,boxstyle="round,pad=0.02,rounding_size=0.12",fc=cols[i],ec="none"))
    tc=NAVY if i<2 else "white"
    ax.text(x+0.92,1.22,t,ha="center",color=tc,fontsize=11.5,fontweight="bold")
    ax.text(x+0.92,0.72,s,ha="center",color=tc,fontsize=8.8)
save(fig,"fig_softmax.png")

# 3 dataflow cycles
hbar("fig_cycles.png","Cycles per pass at N = d = 16, P = 4",["Unfused (baseline)","Fused streaming","Causal prefill","Decode step (LEN 16)"],
     [411,111,97,50],lambda v:f"{v}",[GREY,MID,SKY,NAVY],xlabel="cycles")

# 4 layout progress
fig,ax=plt.subplots(figsize=(7.2,3.0))
names=["sram_p2_v2","dupreg","speed","speed2","final2","final4","final6"]
wns=[-3.75,-2.34,-2.31,-0.342,1.27,-0.32,0.14]
cols=[NAVY if v>=0 else LIGHT for v in wns]; cols[-1]=MID
ax.bar(names,wns,color=cols,width=0.6); ax.axhline(0,color=NAVY,lw=1)
for i,v in enumerate(wns): ax.text(i,v+(0.12 if v>=0 else -0.12),f"{v:+.2f}",ha="center",va="bottom" if v>=0 else "top",fontsize=10.5,fontweight="bold",color=INK)
ax.set_ylabel("setup slack at 50 MHz, tt (ns)"); ax.set_ylim(-4.4,1.9); ax.yaxis.grid(True,color="#E3E9F2"); ax.set_axisbelow(True)
ax.set_title("Typical-corner slack across the layout runs")
fig.text(0.01,-0.03,"final2 had features 1–14; final4 and final6 carry all 17 features. final6 is the sign-off layout.",fontsize=9,color=GREY)
save(fig,"fig_progress.png")

# 5 final6 corners
hbar("fig_corners.png","Setup slack by corner, chip_p2_final6 at 50 MHz",["tt, nominal RC","tt, min RC","tt, max RC","ff, nominal RC","ff, max RC"],
     [0.14,0.954,-0.67,6.13,5.99],lambda v:f"{v:+.2f} ns",[MID,MID,LIGHT,NAVY,NAVY],xlim=(-1.6,7.4),
     note="ss corners run at about 26 MHz: the OpenRAM macro has only a typical-corner timing model.")

# 6 accuracy (Tiny Shakespeare)
lab=["MF2 arithmetic (default)","Exact exponent","Key skip, SKIPTHR 6","Max-first decode","Edge point (K8, V4, skip)","K INT8, V MXFP4","K and V MXFP4","K and V INT4"]
val=[0.07,0.15,0.15,0.13,0.72,0.64,2.61,3.33]
hbar("fig_accuracy.png","Perplexity cost vs FP32 (Tiny Shakespeare char LM)",lab,val,lambda v:f"+{v:.2f}%",
     [NAVY,MID,MID,MID,SKY,SKY,LIGHT,LIGHT],xlabel="perplexity increase (%)",figsize=(7.2,3.4))

# 7 v reads
fig,ax=plt.subplots(figsize=(6.4,2.9)); import numpy as np
x=np.arange(2); wd=0.34
a=ax.bar(x-wd/2,[86,84],wd,color=LIGHT,label="Normal decode"); b=ax.bar(x+wd/2,[36,31],wd,color=NAVY,label="Max-first decode")
for bars in (a,b):
    for r in bars: ax.text(r.get_x()+r.get_width()/2,r.get_height()+1.5,f"{int(r.get_height())}%",ha="center",fontweight="bold",color=INK)
ax.set_xticks(x); ax.set_xticklabels(["Tiny Shakespeare","WikiText-2"]); ax.set_ylim(0,112); ax.set_yticks([0,20,40,60,80,100]); ax.set_ylabel("keys needing a V read (%)")
ax.yaxis.grid(True,color="#E3E9F2"); ax.set_axisbelow(True); ax.legend(frameon=False,loc="upper center",ncol=2,bbox_to_anchor=(0.5,1.02))
ax.set_title("Max-first decode skips most value reads")
save(fig,"fig_vreads.png")

# 8 story ppl
hbar("fig_story.png","Perplexity on the 8 longest TinyStories stories (positions ≥ 32)",
     ["FP32, full context","FP32, 32-token window","Chip, 16-token INT8 window","Chip, 32-token MXFP4 window","Chip, 32-token MXFP4 H2O"],
     [18.36,23.07,36.99,57.42,63.40],lambda v:f"{v:.1f}",[GREY,GREY,NAVY,MID,SKY],xlabel="perplexity (lower is better)")

# 9 power donut
fig,ax=plt.subplots(figsize=(6.0,3.0))
vals=[28.2,18.5,15.9,7.7]; labs=["Combinational","Sequential","Clock tree","SRAM macros"]
w,_=ax.pie(vals,colors=[NAVY,MID,SKY,LIGHT],startangle=90,counterclock=False,wedgeprops=dict(width=0.38,edgecolor="white",linewidth=2))
ax.text(0,0.08,"70.3 mW",ha="center",va="center",fontsize=17,fontweight="bold",color=NAVY); ax.text(0,-0.2,"fused pass",ha="center",color=GREY,fontsize=10)
ax.legend(w,[f"{l}  {v} mW" for l,v in zip(labs,vals)],frameon=False,loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=11)
ax.set_title("Gate-level power, typical corner")
save(fig,"fig_power.png")

# 10 comparison
hbar("fig_compare.png","Energy efficiency at 130 nm (GOP/J)",["A3 (HPCA 2020)","SpAtten 1/8 (HPCA 2021)","Raijin (this work)","ELSA (ISCA 2021)"],
     [31,44,67,84],lambda v:f"{v}",[LIGHT,LIGHT,NAVY,LIGHT],figsize=(7.2,2.6),
     note="Prior designs: 40 nm synthesis or layout, scaled to 130 nm. Raijin: post-layout, gate-level power. None of the prior three has a KV cache.")

# 11 ablation speedups
hbar("fig_ablation.png","What each LLM feature buys (GQA layer, 512-token prompt)",["Prefill time (causal prefill)","Decode rate (GQA decode)","V reads (INT4 V + key skip)"],
     [4.1,4.0,5.3],lambda v:f"{v:.1f}×",[MID,NAVY,SKY],figsize=(7.2,2.3),xlabel="improvement over the baseline")
print("ok")

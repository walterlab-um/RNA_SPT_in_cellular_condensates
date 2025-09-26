# ------------------------------------------------------------------
# 0.  Imports & user paths
# ------------------------------------------------------------------
import os, ast, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tifffile import imread
from shapely.geometry import Polygon
from rich.progress import track
from matplotlib.patches import Rectangle
from shapely.geometry import Point, Polygon

ROOT          = "/Users/esumrall/Desktop/RNA-in-HOPS_condensates/4h/FL-2x"
VIDEO_FILE    = "Merged.tif"
TRACKS_FILE   = "SPT_results_AIO-20220508-FLmRNA_2x_FOV-6-right_reformatted.csv"
COND_FILE     = "condensates_AIO-20220508-FLmRNA_2x_FOV-6-cropped-left.csv"
CELLFILE      = "20220508-FLmRNA_2x_FOV-6-cell-1.txt"

UM_PER_PX     = 0.117
NM_PER_PX     = UM_PER_PX*1000
S_PER_FRAME   = 0.1
# INTERACTION_PX= 1

os.chdir(ROOT)

# ------------------------------------------------------------------
# 1.  Helper functions
# ------------------------------------------------------------------
def parse_list(s):               # '[1,2,3]' -> [1,2,3]
    return list(map(float, s[1:-1].split(', ')))

def wide2long(df):
    rows=[]
    for _,r in df.iterrows():
        for t,x,y in zip(parse_list(r.list_of_t),
                         parse_list(r.list_of_x),
                         parse_list(r.list_of_y)):
            rows.append([int(r.trackID),t,x,y])
    return pd.DataFrame(rows,columns=['trackID','t','x','y'])

def str2poly(s):
    pts=[tuple(map(float,p.split(', ')))
         for p in s[2:-2].split('], [')]
    return Polygon(pts)

cell_xy = np.loadtxt(CELLFILE, dtype=float)                     #  << NEW
print(f"Cell boundary points : {cell_xy.shape[0]}")

# ------------------------------------------------------------------
# 2.  Load data
# ------------------------------------------------------------------
video   = imread(VIDEO_FILE)
df_wide = pd.read_csv(TRACKS_FILE)
df_long = wide2long(df_wide)
df_cond = pd.read_csv(COND_FILE)
print(f"Video  : {video.shape}")
print(f"Tracks : {len(df_wide)} (wide)  {len(df_long)} (rows long)")
print(f"Cond   : {len(df_cond)} rows")

# ------------------------------------------------------------------
# 3.  COLocalization  (per-position flags)
# ------------------------------------------------------------------
col_rows = []
for tid in track(df_wide.trackID, description="Colocalizing"):
    w  = df_wide[df_wide.trackID == tid].iloc[0]
    xs, ys, ts = map(parse_list,
                     [w.list_of_x, w.list_of_y, w.list_of_t])

    for x, y, t in zip(xs, ys, ts):
        inside = False
        cid    = np.nan

        # test against *every* condensate present in that frame
        for _, cr in df_cond[df_cond.frame == t].iterrows():
            poly = str2poly(cr.contour_coord)
            if poly.contains(Point(x, y)):
                inside = True
                cid    = cr.condensateID
                break            # first hit is enough

        col_rows.append([tid, t, x, y, inside, cid])

df_col = pd.DataFrame(col_rows, columns=[
    "trackID", "t", "x", "y", "InCond", "condID"
])
df_col.to_csv("colocalization_positions.csv", index=False)
print(f"Colocalization table written → {df_col.InCond.sum()} TRUE frames")


# master set of dwelling tracks (≥1 True anywhere)
DWELL_IDS = set(df_col[df_col.InCond].trackID.unique())
print(f"{len(DWELL_IDS)} tracks contain at least one In-Condensate frame")

# ------------------------------------------------------------------
# 4.  Robust dwell statistics  (contiguous True blocks)
# ------------------------------------------------------------------
stats=[]
for tid in track(sorted(DWELL_IDS),description="dwell stats"):
    d   = df_col[df_col.trackID==tid].sort_values('t')
    vec = d.InCond.to_numpy()
    n_dwell,flag=0,False
    for b in vec:
        if b and not flag: n_dwell+=1
        flag=b
    ttl   = vec.sum()*S_PER_FRAME
    frac  = vec.mean()
    stats.append([tid,len(vec),n_dwell,ttl,frac,d.x.mean(),d.y.mean()])
st_cols=['trackID','tracklen','N_dwell','TTL_s','frac_dwell','mx','my']
df_stats=pd.DataFrame(stats,columns=st_cols)
df_stats.to_csv("robust_dwell_stats_output.csv",index=False)

# helper dict for quick legend look-up
STAT_LOOK = df_stats.set_index('trackID').to_dict('index')

# ------------------------------------------------------------------
# 5.  Global reconstruction (only DWELL_IDS)
# ------------------------------------------------------------------
lastF=int(df_long.t.max())+1
h,w = video[lastF-1,1].shape
fig,ax=plt.subplots(figsize=(10,8))
for _,cr in df_cond[df_cond.frame==lastF-1].iterrows():
    xs,ys=np.array(str2poly(cr.contour_coord).exterior.xy)
    ax.plot(xs,ys,lw=2,c='#2E86AB')

ax.plot(cell_xy[:,0], cell_xy[:,1], c='k', lw=1)

# all dwelling tracks
for tid in DWELL_IDS:
    d = df_long[df_long.trackID == tid]
    ax.plot(d.x, d.y, c='#F24236', lw=1.4, alpha=0.7)

ax.set(xlim=(0,w),ylim=(h,0),aspect='equal'); ax.axis('off')

bar_px_10um = 10 / UM_PER_PX        # convert 10 µm → pixels
bar_thick   = 6                     # bar height in px
margin      = 20                    # offset from image edge

x0 = w - bar_px_10um - margin       # lower-left-corner of bar
y0 = h - margin - bar_thick

ax.add_patch(Rectangle((x0, y0), bar_px_10um, bar_thick,
                       facecolor='black', edgecolor='white', lw=1.2))
ax.text(x0 + bar_px_10um/2, y0 - 10, '10 μm',
        ha='center', va='top', fontsize=12, fontweight='bold', color='black',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8))

plt.tight_layout(); plt.savefig("global_dwelling_tracks.png",dpi=300)

# ------------------------------------------------------------------
# 6.  Zoomed panels
# ------------------------------------------------------------------
print("Building zoom panels …")
COLORS=['#F24236','#E66100','#D95F02','#CC79A7',
        '#56B4E9','#0072B2','#009E73','#A63603']

borders=df_cond[df_cond.frame==lastF-1]
for _,cr in borders.iterrows():
    cid=int(cr.condensateID)
    members=set(df_col.loc[(df_col.condID==cid)&df_col.InCond,'trackID'])
    members &= DWELL_IDS              # only those flagged dwelling
    if not members: continue

    poly = str2poly(cr.contour_coord)
    xs,ys=np.array(poly.exterior.xy)
    cxmin,cxmax=xs.min()-30,xs.max()+30
    cymin,cymax=ys.min()-30,ys.max()+30
    cxmin=max(0,cxmin); cxmax=min(w,cxmax)
    cymin=max(0,cymin); cymax=min(h,cymax)

    fig,ax=plt.subplots(figsize=(6,6))
    ax.plot(xs,ys,c='#2E86AB',lw=3,label=f'Cond {cid}')
    ax.plot([xs[-1],xs[0]],[ys[-1],ys[0]],c='#2E86AB',lw=3)
    for i,tid in enumerate(sorted(members)):
        d=df_long[df_long.trackID==tid]
        st=STAT_LOOK[tid]
        lbl=(f"Track {tid}\nN={st['N_dwell']}"
             f"  T={st['TTL_s']:.1f}s\nf={st['frac_dwell']:.2f}")
        ax.plot(d.x,d.y,c=COLORS[i%len(COLORS)],lw=2,
                alpha=0.85,label=lbl)
        ax.plot(d.x.iloc[0],d.y.iloc[0],'o',
                mfc='white',mec=COLORS[i%len(COLORS)],ms=5)

        bar_px_1um  = 1 / UM_PER_PX         # 1 µm → pixels
        bar_thick_z = 3                     # bar height
        scale_x = cxmax - bar_px_1um - 10   # 10 px inset from right edge
        scale_y = cymax - 10 - bar_thick_z  # 10 px up from bottom

        ax.add_patch(Rectangle((scale_x, scale_y),
                            bar_px_1um, bar_thick_z,
                            facecolor='black', edgecolor='white', lw=1))
        ax.text(scale_x + bar_px_1um/2, scale_y - 8, '1 μm',
                ha='center', va='top', fontsize=10, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))


    ax.set_xlim(cxmin,cxmax); ax.set_ylim(cymax,cymin)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"Condensate {cid} – {len(members)} RNAs",
                 fontsize=14,weight='bold')
    ax.legend(loc='upper left',fontsize=7,framealpha=0.9)
    plt.tight_layout()
    fname=f"zoom_cond_{cid:02d}.png"
    plt.savefig(fname,dpi=300); plt.close()
    print(f"   saved {fname}")

print("✅ All plots complete.")

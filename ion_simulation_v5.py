"""50×50 Na/K 문자 격자 실험. Python 표준 라이브러리만 사용.
실행: python ion_simulation_v5.py  /  검증: python ion_simulation_v5.py --self-test
설정은 아래 상수를 변경하세요. 실제 막전위·전기력·ATP 반응은 계산하지 않습니다.
"""
import random
import sys
from collections import Counter
from dataclasses import dataclass

WIDTH = HEIGHT = 50
MID = WIDTH // 2
STEP_MS = 10  # 물리 이동 100회/초, 화면 갱신은 약 30회/초
NA_SECONDS = 15
K_SECONDS = 18
# 초기 이온 수: (세포 안 Na, 세포 안 K, 세포 밖 Na, 세포 밖 K)
INITIAL_COUNTS = (125, 375, 375, 125)
SEED = None  # 매번 새로운 난수. 재현하려면 42 같은 정수로 변경.
# 0~47행에 Na/K/펌프를 교대로 배치, 마지막 48/49행은 Na/K.
NA_ROWS = tuple(range(0, HEIGHT, 3))
K_ROWS = tuple(range(1, HEIGHT, 3))
PUMP_ROWS = tuple(range(2, HEIGHT-1, 3))
LEAK_ROW = K_ROWS[0]
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))

@dataclass
class Pump:
    y: int
    phase: str = 'Na'
    cycles: int = 0
    bound: tuple = ()  # 실제 결합한 이온의 격자 좌표; 다음 수송까지 확산 금지

class Simulation:
    def __init__(self, stage=1, seed=SEED, counts=INITIAL_COUNTS):
        if stage not in (1, 2, 3, 4):
            raise ValueError('stage must be 1..4')
        self.stage = stage
        self.rng = random.Random(seed)
        self.grid = {}
        self.ticks = 0
        self.na_open = False
        self.k_open = {LEAK_ROW} if stage >= 3 else set()
        self.sequence = None
        self.pumps = [Pump(y) for y in PUMP_ROWS] if stage == 4 else []
        self.na_out = self.k_in = 0
        self.channel_passages = Counter()
        self.time = 0.0
        self.next_move = STEP_MS / 1000
        self.deadline = None
        self.pump_enabled = stage == 4
        self.history = []
        self.events = []
        if len(counts) != 4 or any(not isinstance(n, int) or n < 0 for n in counts):
            raise ValueError('이온 개수는 0 이상의 정수 4개로 설정하세요.')
        if stage == 1:
            # 좌우 농도 조건을 사용하지 않고 전체 공간에서 한 번에 섞습니다.
            regions = [(range(WIDTH), counts[0] + counts[2], counts[1] + counts[3])]
        else:
            regions = [(range(MID), counts[0], counts[1]),
                       (range(MID, WIDTH), counts[2], counts[3])]
        for xs, na, k in regions:
            cells = [(x, y) for x in xs for y in range(HEIGHT)]
            if na + k > len(cells):
                raise ValueError(f'해당 영역의 이온 수 합은 {len(cells)} 이하여야 합니다.')
            self.rng.shuffle(cells)
            for pos in cells[:na]:
                self.grid[pos] = 'Na'
            for pos in cells[na:na+k]:
                self.grid[pos] = 'K'

        self.record()
        if stage >= 3:
            self.event('K', '초기 개방 (1/17)')
        if stage == 4:
            self.event('Pump', '개방 (시작)')

    def record(self):
        if self.stage >= 2:
            c = self.counts()
            self.history.append((self.time, c['Na', 'in'], c['K', 'in']))

    def event(self, kind, label):
        if self.stage >= 2:
            self.events.append((self.time, kind, label))

    def set_na(self, opened):
        if self.na_open != opened:
            self.na_open = opened
            self.event('Na', '개방 (17/17)' if opened else '중지 (0/17)')

    def set_k(self, rows):
        rows = set(rows)
        if self.k_open != rows:
            before = len(self.k_open)
            self.k_open = rows
            count = len(rows)
            label = '개방' if count > before else '중지'
            self.event('K', f'{label} ({count}/17)' + (' · 1개 유지' if count == 1 else ''))

    def toggle_na(self):
        if self.stage >= 3 and self.sequence is None:
            self.set_na(not self.na_open)

    def toggle_k(self):
        if self.stage >= 3 and self.sequence is None:
            self.set_k({LEAK_ROW} if len(self.k_open) == 17 else K_ROWS)

    def toggle_pump(self):
        if self.stage == 4:
            self.pump_enabled = not self.pump_enabled
            self.event('Pump', '개방' if self.pump_enabled else '중지')

    def start_sequence(self):
        if self.stage >= 3 and self.sequence is None:
            self.sequence = 'Na'
            self.deadline = self.time + NA_SECONDS
            self.set_na(True)
            self.set_k(set())

    def allowed(self, old, new, ion):
        x, y = new
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
            return False
        crossing = (old[0] < MID) != (x < MID)
        if not crossing or self.stage == 1:
            return True
        if self.stage == 2:
            return False
        return ((ion == 'Na' and self.na_open and y in NA_ROWS)
                or (ion == 'K' and y in self.k_open))

    def move_ions(self):
        # 모든 이온은 이동 전 배치를 기준으로 방향을 골라 한 틱에 한 번만 확산합니다.
        # 빈칸으로만 이동하며, 같은 목적지를 고르면 무작위로 한 이온만 이동합니다.
        proposals = {}
        locked = {pos for pump in self.pumps for pos in pump.bound}
        for old, ion in self.grid.items():
            if old in locked:
                continue
            dx, dy = self.rng.choice(DIRECTIONS)
            new = (old[0]+dx, old[1]+dy)
            if new not in self.grid and self.allowed(old, new, ion):
                proposals.setdefault(new, []).append(old)
        winners = [(self.rng.choice(origins), new) for new, origins in proposals.items()]
        for old, new in winners:
            ion = self.grid.pop(old)
            self.grid[new] = ion
            if self.stage >= 3 and (old[0] < MID) != (new[0] < MID):
                # 성공한 통로 통과만 집계하며 펌프 수송은 별도로 셉니다.
                self.channel_passages[ion] += 1

    def run_pumps(self):
        if not self.pump_enabled:
            return
        order = self.pumps[:]
        self.rng.shuffle(order)
        for pump in order:
            outward = pump.phase == 'Na'
            source_x = MID-1 if outward else MID
            ion, needed = ('Na', 3) if outward else ('K', 2)
            if not pump.bound:
                source = [(source_x, y) for y in range(pump.y-1, pump.y+2)
                          if self.grid.get((source_x, y)) == ion]
                if len(source) >= needed:
                    self.rng.shuffle(source)
                    pump.bound = tuple(source[:needed])
                # 결합과 수송을 구분: 결합한 이온은 다음 이동 틱부터 수송합니다.
                continue
            # 배출 공간: 반대편의 3×3칸. 공간이 없으면 상태를 유지하고 기다립니다.
            xs = range(MID, MID+3) if outward else range(MID-3, MID)
            dest = [(x, y) for x in xs for y in range(pump.y-1, pump.y+2)
                    if (x, y) not in self.grid]
            if len(dest) < needed:
                continue
            self.rng.shuffle(dest)
            for pos in pump.bound:
                del self.grid[pos]
            for pos in dest[:needed]:
                self.grid[pos] = ion
            pump.bound = ()
            if outward:
                self.na_out += 3
                pump.phase = 'K'
            else:
                self.k_in += 2
                pump.cycles += 1
                pump.phase = 'Na'

    def advance(self, seconds):
        """이동(0.01초)과 자동 개폐(15초/18초)를 독립적인 시간표로 처리."""
        if seconds < 0:
            raise ValueError('seconds must be nonnegative')
        target = self.time + seconds
        while True:
            event_time = min(self.next_move, self.deadline if self.deadline is not None else float('inf'))
            if event_time > target + 1e-9:
                break
            self.time = event_time
            # 같은 시각이면 개폐를 먼저 적용합니다.
            if self.deadline is not None and self.deadline <= event_time + 1e-9:
                if self.sequence == 'Na':
                    self.set_na(False)
                    self.set_k(K_ROWS)
                    self.sequence = 'K'
                    self.deadline = event_time + K_SECONDS
                else:
                    # 4번 실험은 자동 반응 후 처음의 K 누출 통로 하나를 유지합니다.
                    self.set_k({LEAK_ROW} if self.stage == 4 else set())
                    self.sequence = None
                    self.deadline = None
            if self.next_move <= event_time + 1e-9:
                self.move_ions()
                self.run_pumps()
                self.ticks += 1
                self.next_move = (self.ticks + 1) * STEP_MS / 1000
                self.record()
        self.time = target

    def step(self):
        self.advance(STEP_MS / 1000)

    def counts(self):
        result = Counter()
        for (x, _), ion in self.grid.items():
            result[(ion, 'in' if x < MID else 'out')] += 1
        return result

    def membrane(self, y):
        if self.stage == 1:
            return ' '
        if self.stage >= 3:
            if y in NA_ROWS:
                return 'N' if self.na_open else 'n'
            if y in K_ROWS:
                return 'K' if y in self.k_open else 'k'
        if self.stage == 4:
            for p in self.pumps:
                if p.y == y:
                    return ('P' if p.phase == 'Na' else 'Q') if self.pump_enabled else 'p'
        return '|'


COLORS = {'Na': '#265bd7', 'K': '#169d4b', 'Pump': '#dc3838'}


def graph_scene(model, viewport=960):
    """화면과 SVG 저장이 공유하는 그래프 도형. 각 이벤트는 정확한 시간 x에 연결."""
    scene = []
    def line(coords, color='#cbd5e1', width=1):
        scene.append(('line', coords, {'fill': color, 'width': width}))
    def text(x, y, value, color='#273444', anchor='center'):
        scene.append(('text', [x,y], {'text': value, 'fill': color, 'anchor': anchor}))
    if model.stage == 1:
        text(420,150,'1번 자유 확산 실험은 입자 수와 이벤트를 기록하지 않습니다.')
        return scene, max(viewport,850), 360
    left, top, bottom = 80, 55, 315
    duration = max(30, model.time)
    scale = max(12.0, (viewport-left-65)/duration)
    right = left + duration*scale
    max_value = max((max(n,k) for _,n,k in model.history), default=1)
    ystep = max(10, ((max_value//5 + 99)//170)*100)
    ymax = max(ystep*5, 100)
    for value in range(0,ymax+1,ystep):
        y = bottom-(bottom-top)*value/ymax
        line([left,y,right,y])
        text(left-10,y,str(value),anchor='e')
    for second in range(0,int(duration)+1,5):
        x=left+second*scale
        line([x,top,x,bottom], '#edf0f4')
        text(x,bottom+17,str(second))
    line([left,top,left,bottom,right,bottom], '#273444',2)
    text(left,22,'세포 안 입자 수',anchor='w')
    text(right,bottom+40,'시간 (초, 일시정지 제외)',anchor='e')
    text(right-230,22,'Na⁺',COLORS['Na'])
    line([right-285,22,right-255,22],COLORS['Na'],2)
    text(right-110,22,'K⁺',COLORS['K'])
    line([right-160,22,right-135,22],COLORS['K'],2)
    for species,index in [('Na',1),('K',2)]:
        points=[]
        for row in model.history:
            points.extend((left+row[0]*scale,bottom-(bottom-top)*row[index]/ymax))
        if len(points)==2:
            points += [points[0]+1,points[1]]
        if points:
            line(points,COLORS[species],2)
    base = bottom + 80
    # 가까운 이벤트의 글씨는 아래 행으로 옮겨 겹침을 피합니다.
    for kind, caption in [('Na','Na⁺ 통로'),('K','K⁺ 통로'),('Pump','펌프')]:
        text(8,base,caption,COLORS[kind],anchor='w')
        tracks=[]
        for t,k,label in model.events:
            if k != kind:
                continue
            x=left+t*scale
            label_text=f'{label} · {t:.2f}s'
            end_x=x+len(label_text)*8+12
            lane=next((i for i,end in enumerate(tracks) if end < x-8),len(tracks))
            if lane == len(tracks): tracks.append(end_x)
            else: tracks[lane]=end_x
            y=base+lane*25
            line([x,bottom+28,x,y-7],COLORS[kind])
            line([x-3,bottom+34,x,bottom+28,x+3,bottom+34],COLORS[kind])
            text(x+4,y,label_text,COLORS[kind],anchor='w')
        base+=max(1,len(tracks))*25+22
    width=max(right+65, max((xy[0]+len(opt.get('text',''))*8+20
                            for shape,xy,opt in scene if shape=='text' and opt.get('anchor')=='w'),default=0))
    return scene, width, base+20


def save_results(model, filename):
    """CSV 2개와 브라우저에서 열 수 있는 SVG를 저장합니다."""
    import csv
    from pathlib import Path
    from xml.sax.saxutils import escape
    base=Path(filename).with_suffix('')
    with open(str(base)+'_counts.csv','w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f); writer.writerow(['time_seconds','Na_inside','K_inside'])
        writer.writerows((f'{t:.6f}',n,k) for t,n,k in model.history)
    with open(str(base)+'_events.csv','w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f); writer.writerow(['time_seconds','kind','event'])
        writer.writerows((f'{t:.6f}',k,label) for t,k,label in model.events)
    scene,w,h=graph_scene(model)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
         '<rect width="100%" height="100%" fill="white"/>']
    for shape,xy,opt in scene:
        color=opt['fill']
        if shape=='line':
            points=' '.join(f'{xy[i]:.2f},{xy[i+1]:.2f}' for i in range(0,len(xy),2))
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{opt["width"]}"/>')
        else:
            anchor={'w':'start','e':'end','center':'middle'}[opt['anchor']]
            svg.append(f'<text x="{xy[0]}" y="{xy[1]}" fill="{color}" font-family="sans-serif" font-size="12" '
                       f'text-anchor="{anchor}" dominant-baseline="middle">{escape(opt["text"])}</text>')
    svg.append('</svg>')
    Path(str(base)+'.svg').write_text('\n'.join(svg),encoding='utf-8')


def save_png(model, filename):
    """스크롤 밖의 전체 그래프까지 2배 해상도 PNG로 저장. Pillow만 추가로 필요."""
    import os
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont
    scene,w,h=graph_scene(model)
    candidates=[
        Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/'malgun.ttf',
        Path('/System/Library/Fonts/AppleSDGothicNeo.ttc'),
        Path('/System/Library/Fonts/Supplemental/AppleGothic.ttf'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'),
    ]
    # 프로그램 옆에 한국어 폰트 파일을 두어 직접 지정할 수도 있습니다.
    candidates.insert(0,Path(__file__).with_name('NotoSansKR-Regular.ttf'))
    scale=2
    font=None
    for path in candidates:
        if path.is_file():
            try:
                font=ImageFont.truetype(str(path),12*scale)
                break
            except OSError:
                pass
    korean=font is not None
    if font is None:
        try: font=ImageFont.truetype('DejaVuSans.ttf',12*scale)
        except OSError: font=ImageFont.load_default(size=12*scale)
    def label(value):
        if korean: return value
        replacements={
            '세포 안 입자 수':'Ion count inside',
            '시간 (초, 일시정지 제외)':'Time (s, pauses excluded)',
            'Na⁺ 통로':'Na+','K⁺ 통로':'K+',
            '펌프':'Pumps','초기 개방':'Initial open','개방':'Open',
            '중지':'Closed','시작':'start','1개 유지':'1 stays open',
        }
        for a,b in replacements.items(): value=value.replace(a,b)
        return value
    # 영문 대체 또는 시스템 폰트 폭 차이로 오른쪽 설명이 잘리지 않도록 여백 계산.
    for kind,xy,opt in scene:
        if kind=='text' and opt['anchor']=='w':
            w=max(w,xy[0]+font.getlength(label(opt['text']))/scale+20)
    picture=Image.new('RGB',(int(w*scale+1),int(h*scale+1)),'white')
    draw=ImageDraw.Draw(picture)
    for kind,xy,opt in scene:
        if kind=='line':
            points=[(round(xy[i]*scale),round(xy[i+1]*scale)) for i in range(0,len(xy),2)]
            draw.line(points,fill=opt['fill'],width=max(1,round(opt['width']*scale)))
        else:
            draw.text((xy[0]*scale,xy[1]*scale),label(opt['text']),font=font,
                      fill=opt['fill'],anchor={'w':'lm','e':'rm','center':'mm'}[opt['anchor']])
    picture.save(filename,format='PNG',dpi=(144,144))


def launch():
    import time
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    root=tk.Tk()
    root.title('Na⁺ / K⁺ — 50×50 이온 시뮬레이션')
    root.geometry('1300x900')
    root.minsize(860,600)
    model=Simulation()
    paused=False
    previous=time.monotonic()
    last_signature=None
    last_graph_update=0.0
    cell_w,cell_h=18,14
    title=tk.StringVar(); status=tk.StringVar(); gates=tk.StringVar(); metrics=tk.StringVar()
    names={1:'자유 확산',2:'통과 불가능한 세포막',3:'선택적 이온 통로',4:'통로 + Na/K 펌프'}
    ttk.Label(root,textvariable=title,font=('',13,'bold')).pack(pady=7)
    controls=ttk.Frame(root); controls.pack(fill='x',padx=10)
    buttons=ttk.Frame(root); buttons.pack(fill='x',padx=10,pady=5)
    legend=tk.StringVar()
    ttk.Label(root,textvariable=legend).pack(pady=3)
    ttk.Label(root,textvariable=metrics,font=('',11,'bold'),justify='left').pack(pady=6)
    frame=ttk.Frame(root); frame.pack(fill='both',expand=True,padx=10)
    board=tk.Canvas(frame,bg='#101821',highlightthickness=0)
    sy=ttk.Scrollbar(frame,orient='vertical',command=board.yview)
    sx=ttk.Scrollbar(frame,orient='horizontal',command=board.xview)
    board.configure(yscrollcommand=sy.set,xscrollcommand=sx.set)
    sy.pack(side='right',fill='y'); sx.pack(side='bottom',fill='x'); board.pack(fill='both',expand=True)
    cell_items={}; cached={}; wall_items={}; wall_cached={}
    for y in range(HEIGHT):
        for x in range(WIDTH):
            visual_x=x+(x>=MID and model.stage>=2)
            cell_items[x,y]=board.create_text(10+(visual_x+.5)*cell_w,10+(y+.5)*cell_h,
                                               text='·',fill='#354352',font=('Courier New',10))
        wall_items[y]=board.create_text(10+(MID+.5)*cell_w,10+(y+.5)*cell_h,
                                        text=' ',fill='#438dff',font=('Courier New',11,'bold'))
    board.configure(scrollregion=(0,0,(WIDTH+(model.stage>=2))*cell_w+20,HEIGHT*cell_h+20))
    ttk.Label(root,textvariable=status).pack(pady=(5,0))
    ttk.Label(root,textvariable=gates).pack()
    ttk.Label(root,text='0.01초마다 이동 · S 정지/재개 · N/K 통로 · M 자동 개폐 · P 펌프 · G 그래프 · R 초기화 · Esc 종료\n'
              '실험 전환/초기화 전 그래프 창에서 결과를 저장할 수 있습니다. 1번은 기록 제외.').pack(pady=5)
    graph=None; plot=None; graph_items=[]; event_table=None; table_size=0
    follow=tk.BooleanVar(value=True)

    def update_graph():
        nonlocal table_size
        if graph is None or not graph.winfo_exists(): return
        scene,w,h=graph_scene(model,max(900,plot.winfo_width()-10))
        # 도형을 지우지 않고 기존 도형의 좌표와 글씨만 갱신합니다.
        for i,(shape,xy,opts) in enumerate(scene):
            if i<len(graph_items) and graph_items[i][0]!=shape:
                plot.delete(graph_items[i][1]); graph_items[i]=(shape,None)
            if i>=len(graph_items): graph_items.append((shape,None))
            item=graph_items[i][1]
            if item is None:
                item=(plot.create_line(*xy,**opts) if shape=='line'
                      else plot.create_text(*xy,font=('',9),**opts))
                graph_items[i]=(shape,item)
            else:
                plot.coords(item,*xy); plot.itemconfigure(item,state='normal',**opts)
        for shape,item in graph_items[len(scene):]:
            plot.itemconfigure(item,state='hidden')
        plot.configure(scrollregion=(0,0,w,h))
        if follow.get():
            cursor=80+model.time*max(12,(max(900,plot.winfo_width()-10)-145)/max(30,model.time))
            if cursor>plot.winfo_width()-200: plot.xview_moveto(max(0,(cursor-plot.winfo_width()+200)/w))
        if table_size>len(model.events):
            event_table.delete(*event_table.get_children()); table_size=0
        for t,k,label in model.events[table_size:]:
            event_table.insert('', 'end', values=(f'{t:.2f}',{'Na':'Na⁺ 통로','K':'K⁺ 통로','Pump':'펌프'}[k],label),tags=(k,))
        table_size=len(model.events)
        graph.title(f'{model.stage}단계 — 세포 안 이온 수 / {model.time:.1f}초')

    def export():
        if model.stage==1:
            messagebox.showinfo('결과 저장','1번 실험은 기록하지 않습니다.',parent=graph); return
        name=filedialog.asksaveasfilename(parent=graph,title='결과 저장 (SVG + CSV 2개)',defaultextension='.svg',
                  initialfile=f'ion_stage{model.stage}.svg',filetypes=[('SVG graph','*.svg')])
        if name:
            try:
                save_results(model,name)
                messagebox.showinfo('저장 완료','그래프 SVG, 입자 수 CSV, 이벤트 CSV를 저장했습니다.',parent=graph)
            except OSError as exc:
                messagebox.showerror('저장 실패',str(exc),parent=graph)

    def export_png():
        if model.stage==1:
            messagebox.showinfo('PNG 저장','1번 실험은 기록하지 않습니다.',parent=graph)
            return
        try:
            import PIL
        except ImportError:
            messagebox.showinfo('Pillow 설치 필요',
                'PNG 저장에는 Pillow가 필요합니다. 터미널에서 아래 명령을 한 번 실행하세요.\n\npython -m pip install pillow',parent=graph)
            return
        name=filedialog.asksaveasfilename(parent=graph,title='전체 그래프 PNG 저장',
                  defaultextension='.png',initialfile=f'ion_stage{model.stage}.png',
                  filetypes=[('PNG image','*.png')])
        if name:
            try:
                save_png(model,name)
                messagebox.showinfo('저장 완료','시간축과 개방·중지 표시를 포함한 전체 그래프를 PNG로 저장했습니다.',parent=graph)
            except (OSError,ValueError,MemoryError) as exc:
                messagebox.showerror('PNG 저장 실패',str(exc),parent=graph)

    def show_graph():
        nonlocal graph,plot,graph_items,event_table,table_size
        if graph is not None and graph.winfo_exists():
            graph.deiconify(); graph.lift(); return
        graph=tk.Toplevel(root); graph.geometry('1100x790')
        graph.protocol('WM_DELETE_WINDOW',graph.withdraw)
        graph_items=[]; table_size=0
        top=ttk.Frame(graph); top.pack(fill='x',padx=8,pady=5)
        ttk.Button(top,text='PNG 저장',command=export_png).pack(side='left',padx=(0,5))
        ttk.Button(top,text='SVG + CSV 저장',command=export).pack(side='left')
        ttk.Checkbutton(top,text='최근 시점 따라가기',variable=follow).pack(side='left',padx=8)
        ttk.Label(top,text='파랑: Na⁺ 통로 · 초록: K⁺ 통로 · 빨강: 펌프').pack(side='left',padx=10)
        pf=ttk.Frame(graph); pf.pack(fill='both',expand=True)
        plot=tk.Canvas(pf,bg='white',highlightthickness=0)
        vy=ttk.Scrollbar(pf,orient='vertical',command=plot.yview)
        vx=ttk.Scrollbar(pf,orient='horizontal',command=plot.xview)
        plot.configure(yscrollcommand=vy.set,xscrollcommand=vx.set)
        vy.pack(side='right',fill='y'); vx.pack(side='bottom',fill='x'); plot.pack(fill='both',expand=True)
        table_frame=ttk.Frame(graph); table_frame.pack(fill='x',padx=8,pady=5)
        event_table=ttk.Treeview(table_frame,columns=('time','kind','action'),show='headings',height=5)
        for col,label,width in [('time','시간 (초)',100),('kind','대상',110),('action','작동 변화',650)]:
            event_table.heading(col,text=label); event_table.column(col,width=width)
        for k,color in COLORS.items(): event_table.tag_configure(k,foreground=color)
        scroll=ttk.Scrollbar(table_frame,command=event_table.yview)
        event_table.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y'); event_table.pack(fill='x',expand=True)
        graph.bind('<KeyPress>',on_key); graph.bind('<KeyRelease>',release_key)
        graph.bind('<FocusOut>',lambda e: pressed.clear())
        update_graph()

    def draw(force=False):
        nonlocal last_signature,last_graph_update
        title.set(f'{model.stage}단계: {names[model.stage]} — {model.time:.1f}초 — '+('일시정지' if paused else '실행 중'))
        signature=(id(model),model.ticks,len(model.events),paused)
        if not force and signature==last_signature: return
        if last_signature is None or last_signature[0] != id(model):
            for (x,y),item in cell_items.items():
                board.coords(item,10+(x+(x>=MID and model.stage>=2)+.5)*cell_w,10+(y+.5)*cell_h)
            board.configure(scrollregion=(0,0,(WIDTH+(model.stage>=2))*cell_w+20,HEIGHT*cell_h+20))
            for item in wall_items.values():
                board.itemconfigure(item,state='hidden' if model.stage==1 else 'normal')
        legend.set(('막 없는 하나의 공간 — 전체 무작위 배치' if model.stage==1 else
                    '왼쪽: 세포 안 | 오른쪽: 세포 밖') + '     N = Na⁺ (파랑), K = K⁺ (초록)' +
                   ('' if model.stage==1 else '\n중앙 막: N/K 열린 통로 · n/k 닫힌 통로 · P/Q 펌프 Na/K 대기 · p 펌프 정지'))
        last_signature=signature
        for pos,item in cell_items.items():
            ion=model.grid.get(pos)
            if force or cached.get(pos,'unset')!=ion:
                board.itemconfigure(item,text='N' if ion=='Na' else 'K' if ion=='K' else '·',
                                    fill='#75a5ff' if ion=='Na' else '#66e1a6' if ion=='K' else '#354352')
                cached[pos]=ion
        for y,item in wall_items.items():
            char=model.membrane(y)
            if force or wall_cached.get(y)!=char:
                board.itemconfigure(item,text=char,fill=('#e4a2ff' if char in 'PQp' else
                                    '#ffffff' if char in 'NK' else '#438dff'))
                wall_cached[y]=char
        c=model.counts()
        bound_na = sum(bool(p.bound) and p.phase=='Na' for p in model.pumps)
        bound_k = sum(bool(p.bound) and p.phase=='K' for p in model.pumps)
        metrics.set(f"통로 누적 통과 (양방향): Na⁺ {model.channel_passages['Na']}개 · K⁺ {model.channel_passages['K']}개\n"
                    f'펌프 누적 수송: Na⁺ 밖으로 {model.na_out}개 · K⁺ 안으로 {model.k_in}개\n'
                    f'현재 결합 중인 펌프: Na⁺ {bound_na}개 · K⁺ {bound_k}개 / 총 {len(model.pumps)}개')
        if model.stage == 1:
            status.set(f"전체 Na⁺ {c['Na','in']+c['Na','out']}개 · K⁺ {c['K','in']+c['K','out']}개"
                       f' · 총 {len(model.grid)}개 (영역 구분 없음)')
        else:
            status.set(f"Na⁺ 안 {c['Na','in']} / 밖 {c['Na','out']}     K⁺ 안 {c['K','in']} / 밖 {c['K','out']}"
                       f'     총 {len(model.grid)}개     펌프 Na 배출 {model.na_out} / K 유입 {model.k_in}')

        state=f'통로: Na {17 if model.na_open else 0}/17 · K {len(model.k_open)}/17' if model.stage>=3 else '통로 없음'
        if model.stage==4: state+=' | 펌프 '+('작동' if model.pump_enabled else '중지')
        if model.sequence is not None: state+=f' | M: {model.sequence} 개방, {max(0,model.deadline-model.time):.1f}초 남음'
        gates.set(state)
        now=time.monotonic()
        if force or paused or now-last_graph_update >= .2:
            update_graph()
            last_graph_update=now

    def synchronize():
        nonlocal previous
        now=time.monotonic()
        if not paused: model.advance(now-previous)
        previous=now

    def action(key):
        nonlocal model,paused,previous,table_size,last_signature
        synchronize()
        if key=='escape': root.destroy(); return
        if key in ('r','1','2','3','4'):
            model=Simulation(int(key) if key.isdigit() else model.stage)
            previous=time.monotonic()
            if event_table is not None and graph.winfo_exists():
                event_table.delete(*event_table.get_children()); table_size=0
                plot.xview_moveto(0); plot.yview_moveto(0)
            if model.stage>=2: show_graph()
        elif key=='s': paused=not paused
        elif key=='n': model.toggle_na()
        elif key=='k': model.toggle_k()
        elif key=='m': model.start_sequence()
        elif key=='p': model.toggle_pump()
        elif key=='g': show_graph()
        draw()

    def zoom(factor):
        nonlocal cell_w,cell_h
        new_w=max(8,min(24,cell_w*factor)); new_h=max(7,min(20,cell_h*factor))
        cell_w,cell_h=new_w,new_h
        for (x,y),item in cell_items.items():
            board.coords(item,10+(x+(x>=MID and model.stage>=2)+.5)*cell_w,10+(y+.5)*cell_h)
            board.itemconfigure(item,font=('Courier New',max(6,int(cell_h*.7))))
        for y,item in wall_items.items():
            board.coords(item,10+(MID+.5)*cell_w,10+(y+.5)*cell_h)
            board.itemconfigure(item,font=('Courier New',max(7,int(cell_h*.8)),'bold'))
        board.configure(scrollregion=(0,0,(WIDTH+(model.stage>=2))*cell_w+20,HEIGHT*cell_h+20))

    pressed=set()
    def on_key(e):
        key=e.keysym.lower()
        if key in ('s','n','k','m','p','g','r','1','2','3','4','escape'):
            if key not in pressed:
                pressed.add(key); action(key)
            return 'break'
    def release_key(e): pressed.discard(e.keysym.lower())
    for stage in range(1,5):
        ttk.Button(controls,text=f'{stage}. {names[stage]}',command=lambda s=stage: action(str(s))).pack(side='left',padx=3)
    for key,label in [('s','S 정지/재개'),('n','N Na 통로'),('k','K K 통로'),('m','M 자동'),('p','P 펌프'),('g','G 그래프'),('r','R 초기화')]:
        ttk.Button(buttons,text=label,command=lambda k=key: action(k)).pack(side='left',padx=2)
    ttk.Button(buttons,text='확대',command=lambda: zoom(1.15)).pack(side='right')
    ttk.Button(buttons,text='축소',command=lambda: zoom(1/1.15)).pack(side='right')
    root.bind('<KeyPress>',on_key); root.bind('<KeyRelease>',release_key)
    root.bind('<FocusOut>',lambda e: pressed.clear())
    def tick():
        synchronize(); draw(); root.after(30,tick)
    draw(True); root.after(30,tick); root.mainloop()


def self_test():
    import tempfile
    from pathlib import Path
    import xml.etree.ElementTree as ET
    assert len(PUMP_ROWS)==16 and not(set(PUMP_ROWS)&(set(NA_ROWS)|set(K_ROWS)))
    for stage in range(1,5):
        s=Simulation(stage); before=Counter(s.grid.values()); counts=s.counts()
        for _ in range(30): s.step()
        assert Counter(s.grid.values())==before
        assert all(0<=x<WIDTH and 0<=y<HEIGHT for x,y in s.grid)
        assert abs(s.time-.3)<1e-8 and s.ticks==30
        if stage==1: assert s.history==[] and s.events==[]
        else: assert len(s.history)==31
        if stage==2: assert s.counts()==counts
    s=Simulation(3,counts=(0,0,0,0))
    assert not s.allowed((MID-1,NA_ROWS[0]),(MID,NA_ROWS[0]),'Na')
    s.toggle_na(); assert s.allowed((MID-1,NA_ROWS[0]),(MID,NA_ROWS[0]),'Na')
    assert not s.allowed((MID-1,NA_ROWS[0]),(MID,NA_ROWS[0]),'K')
    s.toggle_k(); s.toggle_k(); assert s.k_open=={LEAK_ROW}
    s.start_sequence(); s.advance(14.99); assert s.na_open and not s.k_open
    s.advance(.01); assert not s.na_open and len(s.k_open)==17
    s.advance(17.99); assert len(s.k_open)==17
    s.advance(.01); assert not s.k_open and s.sequence is None
    assert any(abs(t-15)<1e-8 and k=='Na' and '중지' in label for t,k,label in s.events)
    assert any(abs(t-33)<1e-8 and k=='K' and '중지' in label for t,k,label in s.events)
    s=Simulation(4,counts=(0,0,0,0))
    s.start_sequence(); s.advance(14.99)
    assert s.na_open and not s.k_open
    s.advance(.01); assert not s.na_open and s.k_open==set(K_ROWS)
    s.advance(17.99); assert s.k_open==set(K_ROWS)
    s.advance(.01)
    assert not s.na_open and s.k_open=={LEAK_ROW} and s.sequence is None
    assert s.deadline is None and s.pump_enabled
    assert s.allowed((MID-1,LEAK_ROW),(MID,LEAK_ROW),'K')
    assert not s.allowed((MID-1,K_ROWS[1]),(MID,K_ROWS[1]),'K')
    assert any(abs(t-33)<1e-8 and k=='K' and '1/17' in label and '1개 유지' in label
               for t,k,label in s.events)
    s.advance(1); assert s.k_open=={LEAK_ROW}
    s.start_sequence(); s.advance(33); assert s.k_open=={LEAK_ROW}
    s=Simulation(4,counts=(0,0,0,0)); s.pumps=[Pump(20)]
    assert (0,'Pump','개방 (시작)') in s.events
    s.grid={(MID-1,y):'Na' for y in (19,20,21)}
    s.toggle_pump(); s.run_pumps(); assert s.na_out==0
    s.toggle_pump(); s.run_pumps(); assert len(s.pumps[0].bound)==3 and s.na_out==0
    s.move_ions(); assert all(pos in s.grid for pos in s.pumps[0].bound)
    s.run_pumps(); assert s.na_out==3 and s.pumps[0].phase=='K'
    s.grid={(MID+5+i,20):'Na' for i in range(3)}
    s.grid.update({(MID,19):'K',(MID,21):'K'})
    s.run_pumps(); assert len(s.pumps[0].bound)==2
    s.run_pumps(); assert s.k_in==2 and s.pumps[0].phase=='Na'
    assert Counter(s.grid.values())=={'Na':3,'K':2}
    s=Simulation(4); s.advance(1.2); s.toggle_na(); s.advance(1.2); s.toggle_pump()
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'result.svg'; save_results(s,path)
        ET.parse(path)
        try:
            from PIL import Image
        except ImportError:
            pass
        else:
            png_path=Path(folder)/'result.png'
            save_png(s,png_path)
            with Image.open(png_path) as png:
                assert png.format=='PNG' and png.width>=1900 and png.height>700
                assert png.getextrema()!=((255,255),(255,255),(255,255))
            with Image.open(png_path) as png:
                png.verify()
        assert len((Path(folder)/'result_counts.csv').read_text(encoding='utf-8-sig').splitlines())==len(s.history)+1
        assert '중지' in (Path(folder)/'result_events.csv').read_text(encoding='utf-8-sig')
    # 1번은 두 영역의 초기 수를 강제하지 않으며 중앙 경계를 양방향으로 통과.
    a=Simulation(1,seed=123); b=Simulation(1,seed=123)
    assert a.grid==b.grid
    assert a.allowed((MID-1,10),(MID,10),'Na')
    assert a.allowed((MID,10),(MID-1,10),'K')
    assert a.counts()['Na','in'] != INITIAL_COUNTS[0]
    assert Counter(a.grid.values())=={'Na':500,'K':500}
    assert Simulation(1).grid != Simulation(1).grid
    assert len(set(PUMP_ROWS))==16 and all(0<y<HEIGHT-1 for y in PUMP_ROWS)
    assert len(NA_ROWS)==len(K_ROWS)==17
    assert set(NA_ROWS)|set(K_ROWS)|set(PUMP_ROWS)==set(range(HEIGHT))
    assert not(set(NA_ROWS)&set(K_ROWS))
    # 네 가장자리 모든 좌표에서 바깥쪽 이동을 거부하며 원위치를 유지.
    class FixedDirection:
        def __init__(self,d): self.d=d
        def choice(self,seq): return self.d if seq is DIRECTIONS else seq[0]
    for stage in (1,2,3,4):
        for d,positions in [((-1,0),[(0,y) for y in range(HEIGHT)]),
                            ((1,0),[(WIDTH-1,y) for y in range(HEIGHT)]),
                            ((0,-1),[(x,0) for x in range(WIDTH)]),
                            ((0,1),[(x,HEIGHT-1) for x in range(WIDTH)])]:
            edge=Simulation(stage,counts=(0,0,0,0))
            edge.grid={pos:'Na' for pos in positions}; before=edge.grid.copy()
            edge.rng=FixedDirection(d); edge.move_ions()
            assert edge.grid==before
    for ion,y in [('Na',NA_ROWS[0]),('K',K_ROWS[0])]:
        t=Simulation(3,counts=(0,0,0,0)); t.set_na(True); t.set_k(K_ROWS)
        t.grid={(MID-1,y):ion}; t.rng=FixedDirection((1,0)); t.move_ions()
        assert t.channel_passages[ion]==1 and t.grid=={(MID,y):ion}
        t.rng=FixedDirection((-1,0)); t.move_ions()
        assert t.channel_passages[ion]==2 and t.na_out==t.k_in==0
    print('PASS: conservation, membrane, selectivity, 0.01s steps, exact 15/18s gates, events, pump switch, 3:2 transport, SVG/CSV')

if __name__=='__main__':
    if '--self-test' in sys.argv: self_test()
    else: launch()

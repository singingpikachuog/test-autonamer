import os, random, time, multiprocessing, queue
from PIL import Image
from tqdm import tqdm

SRC, BGS, DST = "data/commands/pokemon/pokemon_images", "data/commands/pokemon/pokemon_images/backgounds", "data/commands/pokemon/images"
RESIZE_RATIO = eval('4/4')
bg_paths = [os.path.join(BGS, f) for f in os.listdir(BGS) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

def worker(q, prog, wid):
    bgs = [Image.open(p).convert("RGBA") for p in bg_paths]
    while True:
        try:
            name = q.get(timeout=5)
            if name is None: break
            if not name.lower().endswith((".png", ".jpg", ".jpeg")): prog[wid] += 1; continue
            base, path = os.path.splitext(name)[0], os.path.join(DST, os.path.splitext(name)[0])
            os.makedirs(path, exist_ok=True)
            img = Image.open(os.path.join(SRC, name)).convert("RGBA")
            img = img.resize((int(img.width * RESIZE_RATIO), int(img.height * RESIZE_RATIO)))
            img.save(os.path.join(path, "0.png"))
            flip = img.transpose(Image.FLIP_LEFT_RIGHT)
            flip.save(os.path.join(path, "1.png"))
            for i, bg in enumerate(bgs):
                px, py = random.randint(0, bg.width - img.width), random.randint(0, bg.height - img.height)
                for j, v in enumerate((img, flip)):
                    temp = bg.copy(); temp.paste(v, (px, py), v)
                    temp.save(os.path.join(path, f"bg_{i}_{j}.png"))
            prog[wid] += 1
        except queue.Empty:
            break
        except Exception as e:
            print(f"Error processing {name}: {e}"); prog[wid] += 1

def generate_images():
    os.makedirs(DST, exist_ok=True)
    mgr, q = multiprocessing.Manager(), multiprocessing.Manager().Queue()
    names = [f for f in os.listdir(SRC) if os.path.isfile(os.path.join(SRC, f))]
    total, workers, prog = len(names), max(1, multiprocessing.cpu_count()), mgr.dict({i:0 for i in range(multiprocessing.cpu_count())})
    for n in names: q.put(n)
    for _ in range(workers): q.put(None)
    ps = [multiprocessing.Process(target=worker, args=(q, prog, i)) for i in range(workers)]
    for p in ps: p.start()
    with tqdm(total=total, desc="Augmenting Pokémon", unit="img") as bar:
        last = 0
        while any(p.is_alive() for p in ps):
            done = sum(prog.values())
            if done > last: bar.update(done - last); last = done
            if done >= total: break
            time.sleep(0.1)
    for p in ps:
        p.join(timeout=5)
        if p.is_alive(): p.terminate()

if __name__ == "__main__":
    print("⚡ Generating augmented Pokémon images...")
    generate_images()
    print("✅ Done.")

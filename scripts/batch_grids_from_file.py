import math
import random

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
from modules import images
from modules.shared import state


class Script(scripts.Script):
    def title(self):
        return "Batch Grids From File"

    def ui(self, is_img2img):
        # This checkbox would look nicer as two tabs, but there are two problems:
        # 1) There is a bug in Gradio 3.3 that prevents visibility from working on Tabs
        # 2) Even with Gradio 3.3.1, returning a control (like Tabs) that can't be used as input
        #    causes a AttributeError: 'Tabs' object has no attribute 'preprocess' assert,
        #    due to the way Script assumes all controls returned can be used as inputs.
        # Therefore, there's no good way to use grouping components right now,
        # so we will use a checkbox! :)
        checkbox_txt = gr.Checkbox(label="Show Textbox", value=False)
        file = gr.File(label="File with inputs", type='bytes')
        prompt_txt = gr.TextArea(label="Prompts")
        checkbox_txt.change(fn=lambda x: [gr.File.update(visible = not x), gr.TextArea.update(visible = x)], inputs=[checkbox_txt], outputs=[file, prompt_txt])
        return [checkbox_txt, file, prompt_txt]

    def run(self, p, checkbox_txt, data: bytes, prompt_txt: str):
        if (checkbox_txt):
            lines = [x.strip() for x in prompt_txt.splitlines()]
        else:
            lines = [x.strip() for x in data.decode('utf8', errors='ignore').split("\n")]
        lines = [x for x in lines if len(x) > 0]

        while True:
            img_count = len(lines) * p.n_iter * p.batch_size
            batch_count = math.ceil(img_count / p.batch_size)
            print(f"Will process {img_count} images in {batch_count} batches.")

            p.do_not_save_grid = False
            state.job_count = batch_count

            images_per_prompt = p.n_iter * p.batch_size
            last_images = None
            for prompt in lines:
                p.prompt = [prompt for _ in range(images_per_prompt)]
                p.seed = [int(random.randrange(4294967294)) for _ in range(images_per_prompt)]
                processed = process_images(p)
                grid = images.image_grid(processed.images, p.batch_size)
                processed.images.insert(0, grid)
                #images.save_image(processed.images[0], p.outpath_grids, "batch_grids", prompt=prompt, grid=False, info=grid.info["parameters"], seed=-1, p=p)
                last_images = processed.images
                if state.interrupted:
                    break

            if state.interrupted:
                break

        # Just return the last processed set of images
        return Processed(p, last_images, p.seed, "")

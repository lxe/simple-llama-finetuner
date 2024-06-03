from config import SHARE, MODELS, TRAINING_PARAMS, LORA_TRAINING_PARAMS, GENERATION_PARAMS, SERVER_HOST, SERVER_PORT

import os
import gradio as gr
import random

from trainer import Trainer

LORA_DIR = 'lora'

def random_name():
    fruits = [
        "dragonfruit", "kiwano", "rambutan", "durian", "mangosteen", 
        "jabuticaba", "pitaya", "persimmon", "acai", "starfruit"
    ]
    return '-'.join(random.sample(fruits, 3))

class UI():
    def __init__(self):
        self.trainer = Trainer()

    def load_loras(self):
        loaded_model_name = self.trainer.model_name
        if os.path.exists(LORA_DIR) and loaded_model_name is not None:
            loras = [f for f in os.listdir(LORA_DIR)]
            sanitized_model_name = loaded_model_name.replace('/', '_').replace('.', '_')
            loras = [f for f in loras if f.startswith(sanitized_model_name)]
            loras.insert(0, 'None')
            return gr.Dropdown.update(choices=loras)
        else:
            return gr.Dropdown.update(choices=['None'], value='None')

    def training_params_block(self):
        with gr.Row():
            with gr.Column():
                self.max_seq_length = gr.Slider(
                    interactive=True,
                    minimum=1, maximum=4096, value=TRAINING_PARAMS['max_seq_length'],
                    label="Max Sequence Length", 
                )
                        
                self.micro_batch_size = gr.Slider(
                    minimum=1, maximum=100, step=1, value=TRAINING_PARAMS['micro_batch_size'], 
                    label="Micro Batch Size", 
                )

                self.gradient_accumulation_steps = gr.Slider(
                    minimum=1, maximum=128, step=1, value=TRAINING_PARAMS['gradient_accumulation_steps'], 
                    label="Gradient Accumulation Steps", 
                )

                self.epochs = gr.Slider(
                    minimum=1, maximum=100, step=1, value=TRAINING_PARAMS['epochs'], 
                    label="Epochs",
                )

                self.learning_rate = gr.Slider(
                    minimum=0.00001, maximum=0.01, value=TRAINING_PARAMS['learning_rate'],
                    label="Learning Rate",
                )

            with gr.Column():
                self.lora_r = gr.Slider(
                    minimum=1, maximum=64, step=1, value=LORA_TRAINING_PARAMS['lora_r'], 
                    label="LoRA R",
                )

                self.lora_alpha = gr.Slider(
                    minimum=1, maximum=128, step=1, value=LORA_TRAINING_PARAMS['lora_alpha'],
                    label="LoRA Alpha",
                )
                
                self.lora_dropout = gr.Slider(
                    minimum=0, maximum=1, step=0.01, value=LORA_TRAINING_PARAMS['lora_dropout'],
                    label="LoRA Dropout",
                )

    def load_model(self, model_name, progress=gr.Progress(track_tqdm=True)):
        if model_name == '': return ''
        if model_name is None: return self.trainer.model_name
        progress(0, desc=f'Loading {model_name}...')
        self.trainer.load_model(model_name)
        return self.trainer.model_name

    def base_model_block(self):
        self.model_name = gr.Dropdown(label='Base Model', choices=MODELS)

    def training_data_block(self):
        training_text = gr.TextArea(
            lines=20, 
            label="Training Data", 
            info='Paste training data text here. Sequences must be separated with 2 blank lines'
        )
        
        examples_dir = os.path.join(os.getcwd(), 'example-datasets')

        def load_example(filename):
            with open(os.path.join(examples_dir, filename) , 'r', encoding='utf-8') as f:
                return f.read()
            
        example_filename = gr.Textbox(visible=False)
        example_filename.change(fn=load_example, inputs=example_filename, outputs=training_text)
        
        gr.Examples("./example-datasets", inputs=example_filename)

        self.training_text = training_text

    def training_launch_block(self):
        with gr.Row():
            with gr.Column():
                self.new_lora_name = gr.Textbox(label='New PEFT Adapter Name', value=random_name())
            with gr.Column():
                train_button = gr.Button('Train', variant='primary')
                abort_button = gr.Button('Abort')

        def train(
            training_text, 
            new_lora_name, 
            max_seq_length, 
            micro_batch_size, 
            gradient_accumulation_steps, 
            epochs, 
            learning_rate, 
            lora_r, 
            lora_alpha, 
            lora_dropout, 
            progress=gr.Progress(track_tqdm=True)
        ):
            self.trainer.unload_lora()

            self.trainer.train(
                training_text, 
                new_lora_name, 
                max_seq_length=max_seq_length,
                micro_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                epochs=epochs,
                learning_rate=learning_rate,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )

            return new_lora_name

        train_event = train_button.click(
            fn=train,
            inputs=[
                self.training_text,
                self.new_lora_name,
                self.max_seq_length, 
                self.micro_batch_size, 
                self.gradient_accumulation_steps, 
                self.epochs, 
                self.learning_rate, 
                self.lora_r, 
                self.lora_alpha, 
                self.lora_dropout, 
            ],
            outputs=[self.new_lora_name]
        )

        train_event.then(
            fn=lambda x: self.trainer.load_model(x, force=True),
            inputs=[self.model_name],
            outputs=[]
        )

        def abort(progress=gr.Progress(track_tqdm=True)):
            print('Aborting training...')
            self.trainer.abort_training()
            return self.new_lora_name.value

        abort_button.click(
            fn=abort,
            inputs=None,
            outputs=[self.new_lora_name],
            cancels=[train_event]
        )

    def inference_block(self):
        with gr.Row():
            with gr.Column():
                self.lora_name = gr.Dropdown(
                    interactive=True,
                    choices=['None'],
                    value='None',
                    label='LoRA',
                )              

                def load_lora(lora_name, progress=gr.Progress(track_tqdm=True)):
                    if lora_name == 'None':
                        self.trainer.unload_lora()
                    else:
                        self.trainer.load_lora(f'{LORA_DIR}/{lora_name}')
                    
                    return lora_name

                self.lora_name.change(
                    fn=load_lora,
                    inputs=self.lora_name,
                    outputs=self.lora_name
                )

                self.prompt = gr.Textbox(
                    interactive=True,
                    lines=5,
                    label="Prompt",
                    value="Human: How is cheese made?\nAssistant:"
                )

                self.generate_btn = gr.Button('Generate', variant='primary')
                self.cancel_btn = gr.Button('Cancel', variant='primary')

                with gr.Row():
                    with gr.Column():
                        self.max_new_tokens = gr.Slider(
                            minimum=0, maximum=4096, step=1, value=GENERATION_PARAMS['max_new_tokens'],
                            label="Max New Tokens",
                        )
                    with gr.Column():
                        self.do_sample = gr.Checkbox(
                            interactive=True,
                            label="Enable Sampling (leave off for greedy search)",
                            value=True,
                        )

                       
                with gr.Row():
                    with gr.Column():
                        self.num_beams = gr.Slider(
                            minimum=1, maximum=10, step=1, value=GENERATION_PARAMS['num_beams'],
                            label="Num Beams",
                        )

                    with gr.Column():
                        self.repeat_penalty = gr.Slider(
                            minimum=0, maximum=4.5, step=0.01, value=GENERATION_PARAMS['repetition_penalty'],
                            label="Repetition Penalty",
                        )

                with gr.Row():
                    with gr.Column():
                        self.temperature = gr.Slider(
                            minimum=0.01, maximum=1.99, step=0.01, value=GENERATION_PARAMS['temperature'],
                            label="Temperature",
                        )

                        self.top_p = gr.Slider(
                            minimum=0, maximum=1, step=0.01, value=GENERATION_PARAMS['top_p'],
                            label="Top P",
                        )

                        self.top_k = gr.Slider(
                            minimum=0, maximum=200, step=1, value=GENERATION_PARAMS['top_k'],
                            label="Top K",
                        )

            with gr.Column():
                self.output = gr.Textbox(
                    interactive=True,
                    lines=20,
                    label="Output"
                )
            
            
            def generate(
                prompt,
                do_sample,
                max_new_tokens,
                num_beams,
                repeat_penalty,
                temperature,
                top_p,
                top_k,
                progress=gr.Progress(track_tqdm=True)
            ):
                #Iteratively generate tokens until we either emit max_new_tokens or stop getting new output           
                for i in range(max_new_tokens):
                    output_this_iteration = self.trainer.generate(
                        prompt,
                        do_sample=do_sample,
                        max_new_tokens=1,
                        num_beams=num_beams,
                        repetition_penalty=repeat_penalty,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k
                    )
                    #If we have the same output as last iteration, generation is done
                    if len(prompt) == len(output_this_iteration):
                        break
                    
                    prompt = output_this_iteration
                    yield output_this_iteration
                    
            
            generate_event = self.generate_btn.click(
                fn=generate,
                inputs=[
                    self.prompt,
                    self.do_sample,
                    self.max_new_tokens,
                    self.num_beams,
                    self.repeat_penalty,
                    self.temperature,
                    self.top_p,
                    self.top_k
                ],
                outputs=[self.output]
            )

            self.cancel_btn.click(fn=None, inputs=None, outputs=None, cancels=[generate_event])

    def layout(self):
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    gr.HTML("""<h2>
                    <a style="text-decoration: none;" href="https://github.com/lxe/simple-llama-finetuner">🦙 Simple LLM Finetuner</a>&nbsp;<a href="https://huggingface.co/spaces/lxe/simple-llama-finetuner?duplicate=true"><img 
                    src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&amp;style=flat&amp;logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&amp;logoWidth=14" style="display:inline">
                    </a></h2><p>Finetune an LLM on your own text. Duplicate this space onto a GPU-enabled space to run.</p>""")
                with gr.Column():
                    self.base_model_block()
            with gr.Tab('Finetuning'):
                with gr.Row():
                    with gr.Column():
                        self.training_data_block()
                        
                    with gr.Column():
                        self.training_params_block()
                        self.training_launch_block()

            with gr.Tab('Inference') as inference_tab:
                with gr.Row():
                    with gr.Column():
                        self.inference_block()

            inference_tab.select(
                fn=self.load_loras,
                inputs=[],
                outputs=[self.lora_name]
            )

            self.model_name.change(
                fn=self.load_model, 
                inputs=[self.model_name], 
                outputs=[self.model_name]
            ).then(
                fn=self.load_loras,
                inputs=[],
                outputs=[self.lora_name]
            )
                     
        return demo
    
    def run(self):
        self.ui = self.layout()
        self.ui.queue().launch(show_error=True, share=SHARE, server_name=SERVER_HOST, server_port=SERVER_PORT)
                   
if (__name__ == '__main__'):
    ui = UI()
    ui.run()


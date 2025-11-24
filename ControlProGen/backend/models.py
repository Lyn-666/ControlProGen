import time
from transformers import TrainerCallback

class TqdmSyncCallback(TrainerCallback):
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback

    def on_log(self, args, state, control, **kwargs):
        if self.progress_callback and len(state.log_history) > 0:
            last = state.log_history[-1]

            epoch = int(last.get("epoch", state.epoch or 1))

            # global_step = HF 内部训练步
            step = state.global_step
            
            # max_steps = HF 内部总 steps
            total_steps = state.max_steps

            # 速度信息来自 log
            speed = last.get("train_steps_per_second", 0)

            self.progress_callback(epoch, step, total_steps, speed)

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from peft import LoraConfig,get_peft_model
import torch
import re
from datasets import Dataset
from dataprocess import *
from itertools import chain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_function(examples, tokenizer, max_length):
    inputs = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in examples["src"]]
    targets = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in examples["tgt"]]

    # text = inputs, text_target = targets
    model_inputs = tokenizer(
        inputs,
        text_target=targets,    
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    # labels pad_token_id → -100
    labels = model_inputs["labels"]
    labels = [[(l if l != tokenizer.pad_token_id else -100) for l in seq] for seq in labels]
    model_inputs["labels"] = labels
    return model_inputs 

def run_training(df, training_cfg, progress_callback=None):

    df_pair = df_pair_cvs(df)
    json_pair = csv_json_ice(df_pair)
    max_length = max_len(df) + 2
    epochs = training_cfg["epochs"]
    batch_size = training_cfg["batch_size"]

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50",model_max_length=max_length,legacy=False)

    model_config = AutoConfig.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/prot_t5_xl_uniref50",config=model_config).to(device)
    model.resize_token_embeddings(len(tokenizer))

    # LoRA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    lora_model = get_peft_model(model, lora_cfg).to(device)

    train_ds = Dataset.from_list(json_pair)
    tokenized_train_ds = train_ds.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},remove_columns=train_ds.column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir="../logs/checkpoints",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=4e-5,
        warmup_steps=50,
        gradient_accumulation_steps=2,
        fp16=True,
        logging_steps=1,
        logging_strategy="steps",
        log_level="error",
        disable_tqdm=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        data_collator=default_data_collator,
        processing_class=tokenizer,
        callbacks=[TqdmSyncCallback(progress_callback)]
    )

    trainer.train()
    lora_model.eval()
    return lora_model, tokenizer, max_length

def run_generation(df, lora_model, tokenizer, gen_cfg, max_length, update_ui=None):

    gen_numbers = gen_cfg["num_beams"]
    temperature = gen_cfg["temperature"]
    top_k=gen_cfg['top_k']
    top_k_seqs = top_k_seq(df)


    def token_input(input_text):
        if isinstance(input_text, str):
            # Process single sequence
            processed_input = " ".join(list(re.sub(r"[UZOB]", "X", input_text)))
            inputs = [processed_input]
        else:
            # Process list of sequences
            inputs = inputs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in input_text]
        inputs = tokenizer(inputs, max_length=max_length, truncation=True,padding='max_length', return_tensors="pt").to(device)
        return inputs

    def ggt_tgt(inputs,temperature,top_k):
        with torch.no_grad():
            output_sequences = lora_model.generate(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"],
                num_return_sequences=10, 
                max_new_tokens = max_length-1,
                do_sample=True,    
                temperature=temperature,
                top_k=top_k,
                eos_token_id=tokenizer.eos_token_id
                )

            dd =[]
            for idx, sequence in enumerate(output_sequences):
                decoded_output = tokenizer.decode(sequence, skip_special_tokens=True)
                cleaned_output = ''.join(c for c in decoded_output if c.strip() and c in "ACDEFGHIKLMNPQRSTVWYX ")
                cleaned_output = cleaned_output.replace(" ", "")  # Remove spaces
                dd.append(cleaned_output)
        return dd

    def tgt_iter(temp,top_k):
        inputs = token_input(top_k_seqs)
        seq_gen = ggt_tgt(inputs,temp,top_k)
        existing = set(df['sequence'].values)
        return [seq for seq in seq_gen if seq not in existing]

    # streaming container
    if update_ui:
        update_ui("Starting generation...\n")

    # patience round
    tgt_temp = []
    no_new = 0
    max_rounds = 50
    patience_round = 10

    while len(tgt_temp) < gen_numbers and max_rounds > 0:
        max_rounds -= 1

        new_round = tgt_iter(temperature,top_k)
        before = len(tgt_temp)
        # add UNIQUE sequences
        for seq in new_round:
            if seq not in tgt_temp:
                tgt_temp.append(seq)
                if update_ui:
                    update_ui(seq)
                if len(tgt_temp) >= gen_numbers:
                    break
                
        after = len(tgt_temp)

        if after == before:
            no_new += 1
        else:
            no_new = 0

        if no_new >= patience_round:
            message = f"Early stop: Please increase the temperature!. Generated {len(tgt_temp)} sequences."
            return tgt_temp, message

    message = f"Generated {len(tgt_temp)} sequences."
    return tgt_temp[:gen_numbers], message 
         
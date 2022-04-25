import torch

special_chars_dict = {'DOUBLE_LEFT_TOP': u'\u2554',
                      'DOUBLE_VERTI_PIPE': u'\u2551',
                      'DOUBLE_LEFT_BOTTOM': u'\u255a',
                      'DOUBLE_RIGHT_TOP': u'\u2557',
                      'DOUBLE_RIGHT_BOTTOM': u'\u255d',
                      'DOUBLE_HORIZ_PIPE': u'\u2550',
                      'SINGLE_LEFT_TOP': u'\u250c',
                      'SINGLE_VERTI_PIPE': u'\u2502',
                      'SINGLE_LEFT_BOTTOM': u'\u2514',
                      'SINGLE_RIGHT_TOP': u'\u2510',
                      'SINGLE_RIGHT_BOTTOM': u'\u2518',
                      'SINGLE_HORIZ_PIPE': u'\u2500'}


def print_title(title):
    title_len = len(title)
    print(special_chars_dict['DOUBLE_LEFT_TOP'] +
             "".join([special_chars_dict['DOUBLE_HORIZ_PIPE']] * (58 + title_len)) +
             special_chars_dict['DOUBLE_RIGHT_TOP'])
    print(u'\u2551' + "".join([" "] * 29) + title + "".join([" "] * 29) + u'\u2551')
    print(special_chars_dict['DOUBLE_LEFT_BOTTOM'] +
             "".join([special_chars_dict['DOUBLE_HORIZ_PIPE']] * (58 + title_len)) +
             special_chars_dict['DOUBLE_RIGHT_BOTTOM'])


# https://huggingface.co/docs/transformers/perplexity
def compute_text_avg_perplexity_window(subject_interview, tokenizer, model,  window_size=None):

    encodings = tokenizer(subject_interview, return_tensors='pt')
    # The max_length parameter could be set to the max sequence length that the model can deal with.
    # max_length = model.config.n_positions
    max_length = 50
    if window_size is not None:
        max_length = window_size
    stride = 1

    lls = []
    # for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
    for i in range(1, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)

    return ppl.numpy()
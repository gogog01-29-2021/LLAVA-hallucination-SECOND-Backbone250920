# LLAVA-hallucination-SECOND-Backbone250920
python -m lmms_eval \
  --model llava \
  --vision_tower <pretrained_clip_or_openclip> \
  --stages -2 -1 0 1 \
  --attention_thresholding_type attn_topk \
  --attention_threshold 0.2 \
  --positional_embedding_type bilinear_interpolation \
  --contrastive_alphas 0.25 0.25 0.25 0.25 \
  --image_path /path/to/image.jpg \
  --prompt "Is there a dog in the image?"

---
title: Jerome Powell AI Assistant
emoji: 🏦
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: AI assistant fine-tuned on Jerome Powell's Q&A sessions
---

# 🏦 Jerome Powell AI Assistant

A specialized AI assistant fine-tuned on Federal Reserve Chairman Jerome Powell's Q&A sessions, built using Microsoft's Phi3-Mini model. Ask questions about monetary policy, economics, and Federal Reserve operations to get responses in Jerome Powell's distinctive style.

## 🚀 Features

- **Fine-tuned Model**: Based on Microsoft Phi3-Mini, specialized on Jerome Powell's communication style
- **Interactive Interface**: Clean, modern Gradio interface with advanced settings
- **Educational Focus**: Designed for learning about Federal Reserve operations and monetary policy
- **Responsive Design**: Optimized for both desktop and mobile devices

## 📊 Model Information

- **Base Model**: Microsoft Phi3-Mini
- **Fine-tuning**: Specialized on Jerome Powell Q&A data
- **Model Hub**: [BoostedJonP/powell-phi3-mini](https://huggingface.co/BoostedJonP/powell-phi3-mini)
- **Dataset**: [BoostedJonP/JeromePowell-SFT](https://huggingface.co/datasets/BoostedJonP/JeromePowell-SFT)
- **Repository**: [BigJonP/powell-phi3-sft](https://github.com/BigJonP/powell-phi3-sft)

## 🎯 Usage Examples

Try asking questions like:
- "What factors influence Federal Reserve interest rate decisions?"
- "How does the Fed balance inflation and employment objectives?"
- "What is the role of quantitative easing in monetary policy?"
- "How does the Federal Reserve communicate its policy decisions?"

## ⚙️ Advanced Settings

The interface includes adjustable parameters:
- **Max Response Length**: Control the length of generated responses (64-512 tokens)
- **Number of Beams**: Adjust beam search for quality vs. speed (1-8 beams)
- **Temperature**: Control creativity and randomness (0.1-2.0)

## 🔧 Technical Details

### Dependencies
- `torch>=2.0.0,<2.3.0`
- `transformers>=4.48.0,<4.50.0`
- `accelerate>=0.20.0`
- `bitsandbytes>=0.41.0`
- `gradio>=4.0.0,<5.0.0`
- `safetensors>=0.4.0`

### Model Loading
- Uses `@lru_cache` for efficient model loading
- Optimized for Hugging Face Spaces deployment
- Automatic device mapping (CUDA/CPU)

### Generation Parameters
- Beam search with configurable beams
- Temperature-controlled sampling
- Repetition penalty (1.1)
- Early stopping enabled
- Cache optimization

## 🚨 Important Disclaimer

⚠️ **This AI model provides educational insights based on training data and should not be considered as official Federal Reserve communication or financial advice. Always consult official Fed sources for authoritative information.**

## 👨‍💻 Author

- GitHub: [Jonathan Paserman](https://github.com/BigJonP)
- Model Hub: [BoostedJonP](https://huggingface.co/BoostedJonP)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Related Resources

- [Jerome Powell Press Release Q&A](https://www.kaggle.com/datasets/jonathanpaserman/fed-press-release-text)
- [BoostedJonP/JeromePowell-SFT](https://huggingface.co/datasets/BoostedJonP/JeromePowell-SFT)

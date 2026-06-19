<div align="center">

# 🧠 Multi-Layer Perceptron (MLP) From Scratch

### A Pure NumPy Implementation of Deep Learning Fundamentals with Inverted Dropout

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-darkgreen.svg?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Framework](https://img.shields.io/badge/Framework-From--Scratch-orange.svg?style=flat-square)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)

Language / زبان:  
**[🇬🇧 English Version](#-english-version)** | **[🇮🇷 نسخه فارسی](#-نسخه-فارسی)**

---
</div>

## 🇬🇧 English Version

This repository contains a production-grade, object-oriented implementation of a Multi-Layer Perceptron (MLP) built entirely from scratch using only **NumPy**. The primary objective of this project is to master deep learning from first principles, focusing on backpropagation via the Chain Rule, matrix calculus, and regularization mechanics.

### 🚀 Key Features

* **Dynamic Topology:** Easily configure hidden layers and neuron counts via a dynamic list at initialization.
* **He (Kaiming) Initialization:** Prevents vanishing/exploding gradients by scaling initial weights based on layer dimensions.
* **Vectorized ReLU Activation:** Fully vectorized forward propagation and analytical derivative calculations for high-performance backward passes.
* **Inverted Dropout Regularization:** Implements robust training-phase neuron masking scaled by $1/(1-p)$ to ensure seamless test-phase evaluation without scaling modifications.
* **Mini-batch SGD with Shuffling:** Complete dataset permutation at the start of each epoch to break data order bias and accelerate convergence.

### 📊 Empirical Analysis & Experimental Results

During training over 500 epochs with a learning rate of `0.001` and `0.1` dropout probability, key deep learning phenomena were verified:

> 💡 **Train/Test Loss Inversion ($Test\ MSE < Train\ MSE$):** > With dropout active, the training loss stabilizes at a higher value than the testing loss (e.g., Train: `0.258` vs. Test: `0.241`). This confirms correct phase conditioning; the network is artificially constrained during training but operates at full capacity during evaluation.

> 📉 **Overfitting Mitigation:**
> The regularized network achieved a significantly lower test error compared to the non-regularized baseline, empirically proving the generalization power of our custom dropout layer.

### 📂 Architecture & Core Methods

The implementation is cleanly decoupled inside the main Jupyter Notebook (`.ipynb`):
* `forward(x, training=True)`: Manages data flow, applies activations, and caches dropout masks.
* `back_propagation(y_target)`: Executes the step-by-step matrix chain rule from the output layer back to the input.
* `update_weights(lr)`: Updates weights and biases using computed gradients.
* `train(epochs, batch_size, lr)`: Handles the mini-batch slicing, shuffling, and triggers live loss plotting.

---

## 🇮🇷 نسخه فارسی

این مخزن شامل یک پیاده‌سازی شیءگرا و مهندسی‌شده از یک شبکه عصبی چندلایه (MLP) است که از پایه (From Scratch) و تنها با استفاده از کتابخانه **NumPy** توسعه یافته است. هدف اصلی این پروژه، تسلط بر ریاضیات عمیق پس‌انتشار خطا (Backpropagation) بر پایه قاعده زنجیره‌ای (Chain Rule)، حساب ماتریسی و تکنیک‌های منظم‌سازی است.

### 🚀 قابلیت‌های کلیدی مدل

* **معماری کاملاً پویا:** امکان تعیین تعداد لایه‌ها و نورون‌های پنهان به صورت یک لیست در زمان ساخت شیء.
* **مقداردهی اولیه He:** جلوگیری از پدیده انفجار/محو شدگی گرادیان با تنظیم واریانس وزن‌ها متناسب با ابعاد لایه.
* **اکتیویشن برداری ReLU:** اعمال تابع غیرخطی در فاز رفت و ضرب ماتریسی مشتق تحلیلی دقیق آن در فاز برگشت.
* **ریگولاریزاسیون Inverted Dropout:** اعمال ماسک تصادفی روی نورون‌ها و مقیاس‌دهی با فاکتور $1/(1-p)$ در فاز آموزش جهت حفظ پایداری و خاموشی خودکار در فاز تست.
* **آموزش مینی‌بچ همراه با شافل تصادفی:** هم‌زدن کامل داده‌ها در ابتدای هر ایپاک جهت افزایش تعمیم‌پذیری و شکستن سوگیری ترتیب داده‌ها.

### 📊 تحلیل ریاضی و نتایج تجربی آموزش

در طول ۵۰۰ ایپاک آموزش با نرخ یادگیری `0.001` و شدت دراپ‌اوت `0.1`، پدیده‌های تئوریک زیر به وضوح اثبات شدند:

> 💡 **وارونگی خطا (Test MSE < Train MSE):** > به دلیل روشن بودن دراپ‌اوت در فاز آموزش، خطای آموزش (مثلاً `0.258`) بالاتر از خطای تست (`0.241`) قرار می‌گیرد. این پدیده صحت تئوریک پرچم فازهای شبکه را تایید می‌کند؛ مدل در زمان آموزش عمداً ضعیف می‌شود اما در زمان تست با ۱۰۰٪ ظرفیت پیش‌بینی می‌کند.

> 📉 **کاهش بیش‌برازش (Overfitting):**
> مدل منظم‌شده با دراپ‌اوت به خطای تست کمتری نسبت به مدل ساده دست یافت که نشان‌دهنده بهبود چشمگیر قدرت تعمیم‌پذیری شبکه روی داده‌های ندیده است.

### 📂 ساختار متدهای اصلی پروژه

منطق ریاضی و عملیاتی پروژه بدون شلوغ کردن محیط ریدمی، درون فایل نوت‌بوک (`.ipynb`) پیاده‌سازی شده است:
* `forward(x, training=True)`: مدیریت جریان رفت، اعمال اکتیویشن‌ها و ذخیره ماسک‌های دراپ‌اوت.
* `back_propagation(y_target)`: پیاده‌سازی ماتریسی قاعده زنجیره‌ای گام‌به‌گام از لایه خروجی به سمت ورودی.
* `update_weights(lr)`: به‌روزرسانی پارامترهای وزن و بایاس بر اساس گرادیان‌های محاسبه شده.
* `train(epochs, batch_size, lr)`: مدیریت فرآیند شافل، مینی‌بچ‌ها و رسم نهایی نمودار روند کاهش خطا.

---

## 🗺️ Future Roadmap / نقشه راه آینده

- [ ] **Advanced Optimizers:** Upgrading the weight update mechanism from simple SGD to **Adam** and **RMSprop** from scratch.  
  *(ارتقای متد آپدیت وزن‌ها به الگوریتم‌های پیشرفته Adam و RMSprop از صفر)*
- [ ] **Convolutional Neural Networks (CNN):** Implementing 2D Convolutional and Max Pooling layers using NumPy for image processing tasks.  
  *(پیاده‌سازی لایه‌های کانولوشن دوبعدی و Max Pooling با نام‌پای برای پردازش تصویر)*
# Qdrant Projesi (BBC News)

Bu proje, BBC News verisetini kullanarak metinleri vektöre çeviren ve Qdrant üzerinde benzerlik araması yapan basit bir Python uygulamasıdır.  
Amaç: haber metinlerini vektör veritabanında saklamak ve benzer haberleri bulmaktır.

---

## 🔧 Neler Kullanılıyor?

- **Qdrant**: Vektör veritabanı
- **SentenceTransformers**: Metinleri vektöre çeviren model
- **Datasets / Pandas**: BBC News verisini okumak ve işlemek
- **Loguru**: Loglama (neler olduğunu takip etmek için)

---

## 📁 Klasör Yapısı (özet)

```bash
qdrant_project/
├── main.py            # Uygulamayı çalıştıran dosya
├── config/            # Ayarlar
├── data/              # Veri okuma / ön işleme
├── models/            # Embedding (vektör) modeli
├── database/          # Qdrant bağlantısı ve koleksiyon işlemleri
├── services/          # Arama ve öneri servisleri
└── utils/             # Yardımcı fonksiyonlar ve loglama
```

---

## 📦 Kurulum

1. **Gerekli paketleri yükle**

```bash
pip install -r requirements.txt
```

2. **Qdrant sunucusunu başlat**

Docker ile:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

veya sisteminde Qdrant kuruluysa:

```bash
qdrant
```

3. **Projeyi çalıştır**

```bash
python main.py
```

Bu komut:
- BBC News verisini okur
- Metinleri vektöre çevirir
- Qdrant koleksiyonunu oluşturur / doldurur
- Örnek arama ve loglama işlemlerini çalıştırır

---

## ⚙️ Temel Ayarlar

Ana ayarlar `config/settings.py` dosyasında duruyor. Örneğin:

- Qdrant host / port (`localhost:6333`)
- Kullanılan embedding modeli (varsayılan: `sentence-transformers/all-MiniLM-L6-v2`)
- Batch size
- Log dosya yolu (`logs/qdrant_project.log`)

İstersen bu dosyadan:
- modeli değiştirebilir
- batch boyutunu ayarlayabilir
- log seviyesini (INFO / DEBUG vb.) değiştirebilirsin.

---

## 💡 Ne İşe Yarıyor?

- Metinleri (haber içeriklerini) embedding’e çevirir.
- Qdrant’ta saklar.
- Benzer haberleri bulmak için **benzerlik araması** yapar.
- Tüm önemli adımları log dosyasına ve konsola yazar.

Bu repo, hem Qdrant’ı hem de metin embedding kavramını öğrenmek için basit bir başlangıç projesi olarak düşünülebilir.

---

## 📝 Lisans

Bu proje **MIT Lisansı** ile yayınlanmıştır.

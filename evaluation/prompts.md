# Writing prompts for the real-world test set

These sentences were written for this evaluation (they are not taken from
OpenHand-Synth). Participants copy them by hand, so the ground truth is
known exactly. Mix of capitals, digits and punctuation on purpose.

## Single-line prompts (write each on ONE line)

1. Saya belajar di perpustakaan setiap sore.
2. Dosen menjelaskan materi jaringan saraf tiruan.
3. Kapan tugas ini harus dikumpulkan?
4. Rapat dimulai pukul 09.30 di ruang B204.
5. Tolong kirim catatan kuliah kemarin.
6. Ibu membeli tiga kilogram beras dan dua liter minyak.
7. Ujian tengah semester akan diadakan pada tanggal 14 November.
8. Mesin belajar membutuhkan banyak data yang berkualitas.
9. Terima kasih atas bantuannya!
10. Jakarta, 5 Oktober 2026
11. Nama saya Budi Santoso.
12. Apakah model ini dapat membaca tulisan tangan saya?
13. Hujan turun deras sejak pagi tadi.
14. Kita bertemu di kantin setelah kelas selesai.
15. Algoritma ini lebih cepat daripada metode sebelumnya.

## Multiline prompts (keep the line breaks exactly as shown)

M1.
Hari ini cuaca sangat cerah.
Kami pergi ke taman kota.

M2.
Daftar belanja:
beras, telur, dan minyak goreng.
Jangan lupa membeli sabun.

M3.
Catatan kuliah hari ini membahas
konvolusi, pooling, dan
fungsi aktivasi pada jaringan saraf.

M4.
Selamat pagi, Bu.
Saya ingin bertanya tentang jadwal bimbingan.
Terima kasih atas waktunya.

## Ground-truth format

In `ground_truth.csv`, write line breaks as the two characters `\n`
(backslash + n), e.g. `Hari ini cuaca sangat cerah.\nKami pergi ke taman kota.`

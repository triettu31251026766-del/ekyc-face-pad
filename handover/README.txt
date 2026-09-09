==========================================================
BAN GIAO DU AN — eKYC FACE PAD
==========================================================

Folder nay dung de giao cho sinh vien trong nhom de validate,
nghien cuu va HIEU LOGIC du an. (Repo code moi nguoi da co san —
folder nay chi chua tai lieu + logic, khong copy lai code.)

==========================================================
NOI DUNG FOLDER
==========================================================

TOM_TAT_LOGIC_DU_AN.md
    LOGIC TONG THE cua du an: luong du lieu, tung khoi xu ly,
    cac quyet dinh thiet ke, va duong dan file trong repo de mo
    code that len doi chieu. ==> DOC FILE NAY DAU TIEN NEU
    MUON HIEU DU AN CHAY THE NAO.

documents/
    TECHNICAL_DOCUMENTATION_eKYC_FACE_PAD.md   dac ta ky thuat (doc goc)
    Student_Guide_eKYC_Face_PAD_Project.docx   guide sinh vien cua thay/co
    HUONG_DAN_TRAINING_MODEL_eKYC_FACE_PAD.md  lenh chay tung thi nghiem
    PROGRESS.md                                TIEN DO + viec dang lam
    NHAT_KY_THI_NGHIEM.md                      nhat ky ket qua thuc nghiem day du

paper/
    Paper_Robust_Face_PAD_for_eKYC.docx        BAI BAO KHOA HOC (ban chinh, da audit)

images/
    fig1..fig8 PNG — hinh sinh tu du lieu that:
        fig1 pipeline tong quan | fig2 anh live/spoof that qua cac muc suy giam
        fig3 ROC E01 vs E07 | fig4 F1/ACER theo severity (5 loai)
        fig5 bar chart mean/worst-case | fig6 train/val loss
        fig7-8 ANH CODE MAU cua 2 doan logic chinh
             (eval_degradation.py + robustness.py, comment tieng Anh)

==========================================================
CACH DOC / VALIDATE
==========================================================

1. Doc TOM_TAT_LOGIC_DU_AN.md de hieu logic tong the (co san duong dan
   file trong repo de mo code that).
2. Doc PROGRESS.md + NHAT_KY_THI_NGHIEM.md de hieu tung con so.
3. Mo Paper_Robust_Face_PAD_for_eKYC.docx -> doi chieu voi NHAT_KY + images/.
4. Muon chay lai: mo repo goc, doc README.md, chay
   python -m pytest tests/ -q (193 pass), lam theo HUONG_DAN_TRAINING.

==========================================================
LUU Y QUAN TRONG
==========================================================

- SEED toan du an: 123. Nhan: 0 = bona_fide, 1 = spoof.
- Threshold 0.5 co dinh cho moi thi nghiem (protocol).
- KHONG sua so lieu trong paper neu chua chay lai thi nghiem.
- data/ va results/ nam o repo goc (khong copy vi nang).
- Paper chua co ten tac gia -> can dien truoc khi nop.

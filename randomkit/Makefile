all:
	$(CC) -fPIC randomkit.c -c -o $(OUT_DIR)/randomkit.o
	$(CC) -fPIC distributions.c -c -o $(OUT_DIR)/distributions.o
	$(AR) rcs $(OUT_DIR)/librandomkit.a $(OUT_DIR)/randomkit.o $(OUT_DIR)/distributions.o

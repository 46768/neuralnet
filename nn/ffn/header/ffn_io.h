#ifndef FFN_IO_H
/** \file */
#define FFN_IO_H

#include "ffn.h"
#include "file_io.h"

void ffn_export_model(FFNModel*, FileData*);
FFNModel* ffn_import_model(FileData*);

#endif

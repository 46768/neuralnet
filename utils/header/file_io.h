/** \file */
#ifndef COM_FILE_READER_H
#define COM_FILE_READER_H

#include <stdio.h>

/**
 * \struct FileData
 * \brief A FILE* wrapper with extra data
 */
typedef struct {
    unsigned long size; /**< File size */
    char *filename;     /**< File name */
    FILE *file_pointer; /**< File pointer */
} FileData;

FileData *file_get_read(char *);
FileData *file_get_write(char *);
int file_close(FileData *);

#endif

package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
)

var (
	// ImagesDir is the base dir for the converted images
	ImagesDir = "images"
)

func init() {
	if _, err := os.Stat(ImagesDir); err == nil {
		log.Println("Images dir exists, remove it? N/y")
		var input rune
		fmt.Scanf("%c\n", &input)
		if input != 'Y' && input != 'y' {
			os.Exit(0)
		}

		os.RemoveAll(ImagesDir)
	}
}

func main() {
	folders := []string{"train", "test"}
	for _, folder := range folders {
		err := os.MkdirAll(ImagesDir+"/"+folder, 0755)
		if err != nil {
			log.Fatalln(err)
		}

		files, err := filepath.Glob(folder + "/*.jpg")
		if err != nil {
			log.Println(err)
			continue
		}

		len := len(files)
		for i, file := range files {
			log.Printf("Preprocessing %d of %d", i, len)

			err := preprocessImage(file, ImagesDir+"/"+file)
			if err != nil {
				log.Fatalln(err)
			}
		}
	}
}

package main

import "os/exec"

func preprocessImage(file string, outFile string) error {
	args := []string{
		file,
		"-resize",
		"299x299",
		"-background",
		"black",
		"-gravity",
		"center",
		"-extent",
		"299x299",
		outFile,
	}

	return exec.Command("convert", args...).Run()
}

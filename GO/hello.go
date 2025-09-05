package main

import (
	"fmt"
	"os/exec"
)

func main() {
  fmt.Printf("Hello World\n")
  calc()
  callPython()
}

func callPython() {
	cmd := exec.Command("python", "hello.py")
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("Error executing Python script: %v\n", err)
		return
	}
	fmt.Printf("Output from Python script:\n%s\n", string(out))
	fmt.Print(err)
}
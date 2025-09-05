package main

import (
	"fmt"
	"os/exec"
	//"gocv.io/x/gocv"
)

func main() {
  fmt.Printf("Hello World\n")
  //hellocv()
  callPython()
  fmt.Printf("Done.\n")
}

func callPython() {
	// check file exists
	if _, err := exec.LookPath("python"); err != nil {
		fmt.Println("python が見つかりません。スキップします。")
		return
	}
	// run script if exists
	if _, err := exec.Command("cmd", "/C", "if exist hello.py (echo) else (exit 1)").CombinedOutput(); err != nil {
		// on Windows, use this to check existence; if not found, skip
		fmt.Println("hello.py が見つかりません。スキップします。")
		return
	}
	cmd := exec.Command("python", "hello.py")
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("Python スクリプト実行エラー: %v\n出力:\n%s\n", err, string(out))
		return
	}
	fmt.Printf("Python スクリプト出力:\n%s\n", string(out))
}

package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func main() {
	modelfile := "/tmp/output_graph_optimized.pb"
	if err := filesExist(modelfile); err != nil {
		log.Fatal(err)
	}

	model, err := ioutil.ReadFile(modelfile)
	if err != nil {
		log.Fatal(err)
	}

	// Construct an in-memory graph
	graph := tf.NewGraph()

	// Load the model
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	// Open the csv file
	csv, err := os.OpenFile("submission.csv", os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0666)
	if err != nil {
		log.Fatal(err)
	}
	defer csv.Close()

	// Write header
	csv.Write([]byte("id,label\n"))

	// Loop test files
	for i := 1; i <= 12500; i++ {
		log.Printf("Predicting %d of 12500\n", i)

		file := fmt.Sprintf("images/test/%d.jpg", i)
		tmpfile := os.TempDir() + file

		err := preprocessImage(file, tmpfile)
		if err != nil {
			log.Fatalln(err)
		}
		defer os.Remove(tmpfile)

		// Create a tensor to represent the image
		tensor, err := makeTensorFromImage(tmpfile)
		if err != nil {
			log.Fatal(err)
		}

		// Run a session to retrieve the output of the softmax layer based on the 			// input image tensor
		output, err := session.Run(
			map[tf.Output]*tf.Tensor{
				graph.Operation("Mul").Output(0): tensor,
			},
			[]tf.Output{
				graph.Operation("final_result").Output(0),
			},
			nil)
		if err != nil {
			log.Fatal(err)
		}

		// Retrieve the probability that the image is a dog
		dogProba := output[0].Value().([][]float32)[0][0]

		// write the predicted output to the csv
		csv.Write([]byte(fmt.Sprintf("%d,%f\n", i, dogProba)))
	}
}

// Convert the image in filename to a Tensor suitable as input to the Inception model.
func makeTensorFromImage(filename string) (*tf.Tensor, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(bytes))
	if err != nil {
		return nil, err
	}
	// Construct a graph to normalize the image
	graph, input, output, err := constructGraphToNormalizeImage()
	if err != nil {
		return nil, err
	}
	// Execute that graph to normalize this image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

func constructGraphToNormalizeImage() (graph *tf.Graph, input, output tf.Output, err error) {
	const (
		H, W  = 299, 299     // width and height of the image
		Mean  = float32(128) // Mean value
		Scale = float32(128) // Scale value
		// The values must be the same used to train the inception model
		// I find the right values on the C++ label image example on the tensorflow repository
	)

	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.Div(s,
		op.Sub(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.Cast(s,
						op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func filesExist(files ...string) error {
	for _, f := range files {
		if _, err := os.Stat(f); err != nil {
			return fmt.Errorf("unable to stat %s: %v", f, err)
		}
	}
	return nil
}

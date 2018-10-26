import glove
import weat

if __name__ == "__main__":
  weat_tests = ["weat1","weat2","weat3","weat4"]
  #for wt in weat_tests:
  #  glove.create_weat_vec_files("../tests/"+wt+".txt")
  weatname = weat_tests[0]
  print("Running weat test: ", weatname)
  A, B, X, Y = weat.load_weat_test(weatname)
  weat.run_test(A,B,X,Y)

  print("Running elmo weat test: ", weatname)
  A, B, X, Y = weat.load_elmo_weat_test(weatname)
  weat.run_test(A,B,X,Y)
  

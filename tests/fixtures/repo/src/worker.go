package main
type Server struct { Port int }
func (s *Server) Start() error { return nil }
func NewServer(port int) *Server { return &Server{Port: int} }
